import math
from typing import Optional, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from utils.vae_layers import DecoderResBlock
from utils.distribution import Normal
from model.ddgm import DiffusionPrior
from model.plain_decoder import _Decoder


class _DecoderBlock(nn.Module):
    """
    Single block of the Deep Hierarchical VAE decoder

    Input:   s_enc - features from the encoder, from the same level
             s_dec - feature from the previous decoder level

    Output:  p(z_i | ... )   - prior for the current latent
             q(z_i | x, ...) - variational posterior for the current latent
    """
    def __init__(self,
                 in_channels: int,
                 ch_mult: float,
                 num_blocks_per_scale: int,
                 p_width: int,
                 out_channels: int,
                 z_width: int,
                 upsample: Union[int, None],
                 conv_block_params: dict,
                 top_prior: Union[nn.Module, None] = None,
                 min_logvar: float = -8,
                 num_condition_channels: int = 0,
                 ):
        super().__init__()
        self.__dict__.update(locals())
        # how to model variance (in both q and p)
        self.z_up = nn.Conv2d(z_width, in_channels, 1, padding=0)
        # Define backbone NN
        self.resnet = nn.Sequential(*[
            DecoderResBlock(in_channels,
                            int(in_channels * ch_mult),
                            in_channels if b_num + 1 < num_blocks_per_scale else out_channels,
                            stride=1,
                            use_res=True,
                            zero_last=False,
                            **conv_block_params)
            for b_num in range(num_blocks_per_scale)
        ])
        # Upsample output f the block (if required)
        self.upsample = nn.Upsample(size=upsample, mode='bilinear') \
            if upsample is not None else nn.Identity()

        # define NN to get parameters of the prior
        self.top_prior = top_prior
        if isinstance(top_prior, DiffusionPrior):
            self.q_logvar = nn.Parameter(torch.ones([1, z_width, 1, 1]) * (-6))
        
        self.init_p_net()

    def init_p_net(self):
        if self.top_prior is None:
            self.s_to_p = DecoderResBlock(self.in_channels + self.num_condition_channels,
                                      int(self.in_channels*self.ch_mult),
                                      self.p_width,
                                      stride=1,
                                      use_res=True,
                                      zero_last=True,
                                      **self.conv_block_params)
    def get_logvar(self, lv):
        """
        Given the output of the NN, returns the log-variance of the distribution
        :param lv: output of the NN for the var
        :return:
        """
        return torch.clamp(lv, self.min_logvar, 1)

    def get_q_p_dist(self, s_enc, s_dec, mode, condition):
        raise NotImplementedError

    def forward(self, s_enc, s_dec, mode, t=None, condition=None):
        """
        :param s_enc: [MB, z_width, scale, scale]
        :param s_dec: (s_{i+1}) [MB, in_ch, sc, sc]
        :param mode: train or test
        :return: z sample q an p distributions, deterministic features (s_out) to pass to the next block
        """
        p_dist, q_dist, s_dec = self.get_q_p_dist(s_enc, s_dec, mode, condition)
        assert mode in ['train', 'val', 'test', 'sample']
        # if mode == 'decode':
        if s_enc is not None:  # -> decoding
            z = q_dist.sample()
        else:  # -> sampling
            if self.top_prior is None:
                z = p_dist.sample(t=t)
            else:
                N = s_dec.shape[0]
                z = p_dist.sample(N, t=t)
        s_dec = self.upsample(self.resnet(s_dec + self.z_up(z)))

        if mode == 'sample':
            return p_dist, q_dist, z, s_dec
        return p_dist, q_dist, z, s_dec


class DecoderBlock(_DecoderBlock):
    def __init__(self,
                 in_channels: int,
                 z_width: int,
                 ch_mult: float,
                 out_channels: int,
                 num_blocks_per_scale: int,
                 conv_block_params: dict,
                 upsample: Union[int, None],
                 top_prior: Union[nn.Module, None] = None,
                 min_logvar: float = -8,
                 num_condition_channels:int=0
                 ):
        """
                   -------------- s_dec---------------
                   ↓                |                ↓
          s_enc -→ q                |              p, h
                                    ↓
                        z ~ q if decoder else p
                                    ↓
                              z + s_dec + h
                                    ↓ (resnet)
                                  s_out

        Implements the decoder block from the vdvae paper
        s_enc and s_dec are inputs from the previous blocks (encoder and decoder correspondingly)
        """
        super(DecoderBlock, self).__init__(
            in_channels=in_channels,
            ch_mult=ch_mult,
            num_blocks_per_scale=num_blocks_per_scale,
            p_width=2 * z_width + in_channels,
            out_channels=out_channels,
            z_width=z_width,
            upsample=upsample,
            conv_block_params=conv_block_params,
            top_prior=top_prior,
            min_logvar=min_logvar,
            num_condition_channels=num_condition_channels,
        )
        self.s_to_q = DecoderResBlock(in_channels,
                                      max(int(in_channels*ch_mult), 1),
                                      2*z_width,
                                      stride=1,
                                      use_res=True,
                                      zero_last=True,
                                      **conv_block_params)
    
    def get_q_p_dist(self, s_enc, s_dec, mode, condition):
        if self.top_prior is None:
            # get parameters of the prior
            if condition is not None:
                p_out = self.s_to_p(torch.cat([s_dec,condition], 1))
            else:
                p_out = self.s_to_p(s_dec)
            p_params, h = p_out[:, :2 * self.z_width], p_out[:, 2 * self.z_width:]
            p_mu, p_logvar = torch.chunk(p_params, 2, dim=1)
            p_logvar = self.get_logvar(p_logvar)
            p_dist = Normal(p_mu, p_logvar)
        else:
            p_dist = self.top_prior
            if isinstance(p_dist, Normal):
                p_dist.log_var.clamp_(self.min_logvar, 1)

        # get parameters of the variational posterior
        if s_enc is not None:
            s = (s_enc + s_dec) / math.sqrt(2)
            q_mu, q_logvar = torch.chunk(self.s_to_q(s), 2, dim=1)
            q_logvar = self.get_logvar(q_logvar)

            q_dist = Normal(q_mu, q_logvar)
        else:
            q_dist = None
        # add prior features to the output
        if self.top_prior is None:
            s_dec = s_dec + h
        return p_dist, q_dist, s_dec


class LadderDecoder(_Decoder):
    def __init__(self,
                 num_ch: int,
                 scale_ch_mult: float,
                 block_ch_mult: float,
                 data_ch: int,
                 num_postprocess_blocks: int,
                 likelihood: str,
                 num_mix: int,
                 data_dim: int,
                 weight_norm: bool,
                 batch_norm: bool,
                 padding_mode: str,
                 latent_scales: list,
                 latent_width: list,
                 num_blocks_per_scale: int,
                 activation: str,
                 z_L_prior: dict,
                 start_scale_at_x: bool = False,
                 min_logvar: float = -8,
                 decoder_res_mode: str = '2x3',
                 num_condition_channels: int = 0,
                 ):
        super().__init__()
        self.__dict__.update(locals())
        assert len(latent_width) == len(latent_scales)

        self.conv_block_params = {
            'batch_norm': batch_norm,
            'weight_norm': weight_norm,
            'activation': self.get_activation(activation),
            'padding_mode': padding_mode,
            'mode': decoder_res_mode,
        }
        self.decoder_block_params = {
            'ch_mult': block_ch_mult,
            'num_blocks_per_scale': num_blocks_per_scale,
            'conv_block_params': self.conv_block_params,
            'min_logvar': min_logvar,
            'num_condition_channels': num_condition_channels,
        }
        self._decoder_block = lambda args: DecoderBlock(**args, **self.decoder_block_params)

        self.num_scales = len(latent_scales)
        self.num_latents = sum(latent_scales)
        self.image_size = [data_ch, data_dim, data_dim]
        self.num_ch = [num_ch]
        for i in range(self.num_scales-1):
            self.num_ch += [int(self.num_ch[-1] * scale_ch_mult)]
        # reverse the order of latents: from top (z_L) to bottom (z_1)
        self.num_ch.reverse()
        self.latent_scales.reverse()
        self.latent_width.reverse()
        print('Decoder channels', self.num_ch)
        self.num_p_param = _Decoder.get_num_lik_params(likelihood)

        # create dummy s_L input
        L_dim = data_dim // (2 ** self.num_scales)
        if start_scale_at_x:
            L_dim = (2 * data_dim) // (2 ** self.num_scales)
        self.s_L_shape = (self.num_ch[0], L_dim, L_dim)

        # init the NNs
        if isinstance(z_L_prior, Normal):
            z_L_prior = None
        self.decoder_blocks = self.init_decoder_blocks(z_L_prior)
        all_widths = []
        for w, s in zip(self.latent_width, self.latent_scales):
            all_widths += [w] * s
        self.post_process = self.init_post_process(all_widths)

        self.z_to_features = nn.Conv2d(self.latent_width[0], self.num_ch[-1], kernel_size=1)
        self.decoder_blocks[-1].resnet = nn.Identity()

        self.init()
        self.device = None
        self.context_blocks = self.init_context_blocks()

    def forward(self,
                encoder_s: list,
                N: Optional[int] = None,
                t: Optional[float] = None,
                mode: str = 'train',
                ):
        """
        Decoder of the ladder VAE
        :param encoder_s: list of deterministic features from encoder
                           in the bottom-up order [q_1, ..., q_L]
        :param N: number of samples from p(z_L)
        :param t: temperature
        :return: tuple(p_xz_parameters, p_dist, q_dist, z_samples):
            parameters of the conditional generative distribution p(x|{z}),
            list of prior distributions
            list of posterior distributions (or None)
            list of latent variables z
        """
        encoder_s.reverse()  # [s_enc_L, s_enc_{L-1}, ..., s_enc_1]
        # init s_dec
        if encoder_s[0] is not None:  # -> reconstruction
            N = encoder_s[0].shape[0]
        if self.device is None:
            self.device = self.parameters().__next__().device
        s_dec = torch.ones((N,) + self.s_L_shape, device=self.device)

        p_dist = []
        q_dist = []
        z_list = []
        for i, dec_block in enumerate(self.decoder_blocks):
            s_enc = encoder_s[i]   
            
            condition = self.get_condition(z_list, i)
            p, q, z, s_dec = dec_block(s_enc, s_dec, mode, t=t, condition=condition)    
            p_dist.append(p)
            q_dist.append(q)
            z_list.append(z)

        if isinstance(s_dec, tuple):
            s_dec = s_dec[0]
        h = self.final_deterministic(z_list, s_dec)
        p_xz_params = self.get_p_xz_params(h)
        return p_xz_params, p_dist, q_dist, z_list
    
    def final_deterministic(self, z_list, s_dec):
        # Likelihood depends on the aggregated latent variables, not s_dec
        up = nn.Upsample(size=s_dec.shape[-1], mode='bilinear')
        
        const = float(math.sqrt(len(z_list)))
        s_dec = torch.stack([up(z) for z in z_list], 0).sum(0) / const
        s_dec = self.z_to_features(s_dec)

        out_feature = self.post_process(s_dec)

        return out_feature

    def get_condition(self, z_list, i):
        return None

    def get_p_xz_params(self, h) -> tuple:
        if 'logistic_mixture' in self.likelihood:
            log_probs, ll = h[:, :self.num_mix], h[:, self.num_mix:]
            ll = ll.reshape(-1, self.image_size[0], self.num_mix*self.num_p_param, self.image_size[1], self.image_size[2])
            return (log_probs, ) + torch.chunk(ll, self.num_p_param, dim=2)
        else:
            return torch.chunk(h, self.num_p_param, dim=1)

    def z_L_post_proc(self, z_L):
        return z_L

    def init_context_blocks(self):
        context_blocks = nn.ModuleList()
        for scale_id, num_latent in enumerate(self.latent_scales): # latent scales from top (smallest) to bottom (largest)
            scale_size = 2 ** (len(self.latent_scales) - scale_id)
            if self.start_scale_at_x:
                scale_size /= 2
            out_size = self.image_size[-1] // int(scale_size)
            for lat_id in range(num_latent):
                if (scale_id + lat_id) > 0:
                    context_blocks.append(
                        nn.Sequential(
                            nn.Upsample(size=out_size, mode='bilinear'),
                        )
                    )
        return context_blocks
    
    def init_decoder_blocks(self, top_prior) -> nn.ModuleList:
        decoder_backbone = nn.ModuleList()
        z_L_dim = self.s_L_shape[-1]
        scale_sizes = [z_L_dim * (2 ** i) for i in range(1, self.num_scales + 1)]
        scale_sizes[-1] = self.image_size[1]
        for i in range(self.num_scales)[::-1]:
            if self.latent_scales[i] == 0:
                scale_sizes[i-1] = scale_sizes[i]
        for s_num, (s, w) in enumerate(zip(self.latent_scales, self.latent_width)):
            ss = scale_sizes[s_num-1] if s_num > 0 else z_L_dim
            print(f'Scale {self.num_scales-s_num}, {s} latents, out shape: {int(ss)}')

            for latent in range(s):
                out_ch = self.num_ch[s_num]
                is_last = latent+1 == s
                if is_last and s_num+1 < len(self.latent_scales):
                    out_ch = self.num_ch[s_num+1]
                block_params = {
                    'in_channels': self.num_ch[s_num],
                    'z_width': w,
                    'out_channels': out_ch,
                    'upsample': scale_sizes[s_num] if is_last else None,
                    'top_prior': top_prior if (s_num + latent) == 0 else None,
                }
                
                decoder_backbone.append(self._decoder_block(block_params))

                # stable init for the resnet
                res_nn = decoder_backbone[-1].resnet[-1]
                res_nn.net[res_nn.last_conv_id].weight.data *= math.sqrt(1. / self.num_latents)
                decoder_backbone[-1].z_up.weight.data *= math.sqrt(1. / self.num_latents)
        return decoder_backbone
        
    def init_post_process(self, all_widths=None) -> nn.Sequential:
        out_ch = self.num_p_param * self.image_size[0]
        if 'logistic_mixture' in self.likelihood:
            out_ch = self.num_mix * (out_ch + 1)

        act_out = nn.Sigmoid() if self.num_p_param == 1 else nn.Identity()

        post_net = []     
        for i in range(self.num_postprocess_blocks):
            post_net.append(
                DecoderResBlock(
                    self.num_ch[-1],
                    max(int(self.num_ch[-1]*self.block_ch_mult), 1),
                    self.num_ch[-1],
                    stride=1,
                    use_res=True,
                    zero_last=True,
                    **self.conv_block_params)
            )
    
        post_net += [
            nn.Conv2d(self.num_ch[-1], out_ch, kernel_size=3, padding=1),
            act_out
        ]
        post_net = nn.Sequential(*post_net)
        return post_net
