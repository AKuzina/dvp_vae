import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from datasets.dct import DCT
from utils.distribution import Normal
from model.ddgm import DiffusionPrior
from model.decoder import LadderDecoder


class _CtxDecoderBlock(nn.Module):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 ):
        super().__init__()
        self.__dict__.update(locals())
        self.ctx_prior = ctx_prior
        
        if isinstance(self.ctx_prior, DiffusionPrior):
            init_val = self.ctx_prior.noise_schedule.gamma_min
            self.q_logvar = nn.Parameter(torch.ones([1] + ctx_size) * init_val)
        else:
            self.q_logvar = nn.Parameter(torch.ones([1] + ctx_size) * (-6))
        
    def get_logvar(self, MB):
        self.q_logvar.data = torch.clamp(self.q_logvar.data, min=-12., max=-5.)
        return self.q_logvar.repeat(MB, 1, 1, 1)

    def forward(self, ctx_val, s_dec, mode, t=None, condition=None):
        """
        :param ctx_val: Analog of s_enc in LadderVAE.
        :param s_dec: Here for compatibility with decoder block interface
        :param mode: train, test
        :param t: temperature
        :return: (p_dist, q_dist, z_sample, s_dec)
        """
        if ctx_val is None:
            q_dist = None
            ctx_val = self.ctx_prior.sample(s_dec.shape[0])
        else:
            ctx_val = self.preprocess_ctx(ctx_val)
            q_dist = Normal(ctx_val, self.get_logvar(ctx_val.shape[0]))
            ctx_val = q_dist.sample(t=t)
        
        if isinstance(self.ctx_prior, DiffusionPrior) and mode is not 'sample':
            if mode == 'test':
                p_dist = self.ctx_prior.eval_is_ll(ctx_val, is_k=1)
            else:
                p_dist = self.ctx_prior.log_prob(ctx_val, mode=mode, reduce_dim=False)
        else:
            p_dist = self.ctx_prior
        
        return p_dist, q_dist, ctx_val, s_dec

    def ctx_to_x(self, ctx):
        assert NotImplementedError

    def x_to_ctx(self, x):
        assert NotImplementedError

    def preprocess_ctx(self, ctx):
        """
        In needed,  precprocess context that was created on the dataset construction stage.
        E.g. for DCT context we will do normalization and (possibly) quantization on this step.
        :param ctx:
        :return:
        """
        return ctx


class DCTDecoderBlock(_CtxDecoderBlock):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 ):
        super(DCTDecoderBlock, self).__init__(
            x_size=x_size,
            ctx_size=ctx_size,
            ctx_prior=ctx_prior,
            max_scale=max_scale,
            next_ch=next_ch,
        )
        self.dct = DCT(x_size[1], x_size[2])

        # DCT scaling parameters
        self.dct_scale = nn.Parameter(torch.zeros(x_size)[None, :, :self.ctx_size[1], :self.ctx_size[1]], requires_grad=False)
        pad = self.x_size[1] - ctx_size[-1]
        self.pad = (0, pad, 0, pad)

    def ctx_to_x(self, ctx):
        # unnormalize
        ctx = ctx * self.dct_scale

        # pad with 0 and invert DCT
        x = self.dct.idct2(F.pad(ctx, self.pad))
        x = 2 * torch.clamp(x, 0, 1) - 1
        return x

    def x_to_ctx(self, x, preprocess=True):
        dct = self.dct.dct2(x)[:, :, :self.ctx_size[1], :self.ctx_size[1]]
        if preprocess:
            dct = self.preprocess_ctx(dct)
        return dct

    def preprocess_ctx(self, y_dct):
        # normalize
        y_dct = y_dct / self.dct_scale
        # exactly [-1, 1]
        y_dct = torch.clamp(y_dct, -1, 1)
        return y_dct


class DownsampleDecoderBlock(_CtxDecoderBlock):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 ):
        super(DownsampleDecoderBlock, self).__init__(
            x_size=x_size,
            ctx_size=ctx_size,
            ctx_prior=ctx_prior,
            max_scale=max_scale,
            next_ch=next_ch,
        )
        self.kernel_size = int(np.ceil(self.x_size[1] / self.ctx_size[1]))
        self.pad_size = int(
            np.ceil((self.kernel_size * self.ctx_size[1] - self.x_size[1]) // 2))
        

    def ctx_to_x(self, ctx):
        x = torch.nn.functional.interpolate(ctx, 
                                            size=self.x_size[1], 
                                            mode='bilinear')
        return x

    def x_to_ctx(self, x, preprocess=True):
        x_dwn = torch.nn.functional.interpolate(
            x, 
            size=self.ctx_size[1], 
            mode='bilinear'
            )
        if preprocess:
            x_dwn = self.preprocess_ctx(x_dwn)
        return x_dwn

    def preprocess_ctx(self, y_dct):
        return y_dct


class LinearCtxDecoderBlock(_CtxDecoderBlock):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 ):
        super(LinearCtxDecoderBlock, self).__init__(
            x_size=x_size,
            ctx_size=ctx_size,
            ctx_prior=ctx_prior,
            max_scale=max_scale,
            next_ch=next_ch,
        )
        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.x_size), np.prod(self.ctx_size)),
            nn.Unflatten(1, self.ctx_size)
        )
        

    def ctx_to_x(self, ctx):
        x = torch.nn.functional.interpolate(ctx, 
                                            size=self.x_size[1], 
                                            mode='bilinear')
        return x

    def x_to_ctx(self, x, preprocess=True):
        x_ctx = self.f(x)
        if preprocess:
            x_ctx = self.preprocess_ctx(x_ctx)
        return x_ctx

    def preprocess_ctx(self, y_dct):
        return y_dct


class ContextLadderDecoder(LadderDecoder):
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
                 ctx_type: str,
                 ctx_size: int,
                 start_scale_at_x: bool = False,
                 min_logvar: float = -8,
                 decoder_res_mode: str = '2x3',
                 ):
        self.__dict__.update(locals())
        self.max_scale = 2 ** len(latent_scales)
        if self.start_scale_at_x:
            self.max_scale /= 2

        super().__init__(
            num_ch=num_ch,
            scale_ch_mult=scale_ch_mult,
            block_ch_mult=block_ch_mult,
            data_ch=data_ch,
            num_postprocess_blocks=num_postprocess_blocks,
            likelihood=likelihood,
            num_mix=num_mix,
            data_dim=data_dim,
            weight_norm=weight_norm,
            batch_norm=batch_norm,
            padding_mode=padding_mode,
            latent_scales=latent_scales,
            latent_width=latent_width,
            num_blocks_per_scale=num_blocks_per_scale,
            activation=activation,
            z_L_prior=None,
            start_scale_at_x=start_scale_at_x,
            min_logvar=min_logvar,
            decoder_res_mode=decoder_res_mode,
            num_condition_channels=data_ch,
        )
        self.init()
        self.z_L_prior = z_L_prior

        # init context decoder block
        self.ctx_size = [data_ch, ctx_size, ctx_size]

        ctx_block_params = {
            'x_size': self.image_size,
            'ctx_size': self.ctx_size,
            'ctx_prior': z_L_prior,
            'max_scale': self.max_scale,
            'next_ch': self.num_ch[0],
        }
        
        ctx_decoder = {
            'dct': DCTDecoderBlock,
            'downsample': DownsampleDecoderBlock,
            'linear': LinearCtxDecoderBlock, 
        }[ctx_type](**ctx_block_params)
            
        # add to the rest of the blocks
        self.decoder_blocks = nn.ModuleList([ctx_decoder, *self.decoder_blocks])
        
        # Latent aggregation
        self.z_to_features = nn.Conv2d(self.ctx_size[0], self.num_ch[-1], kernel_size=1)
        self.decoder_blocks[-1].resnet = nn.Identity()
            
        self.context_blocks = self.init_context_blocks()

    def init_context_blocks(self):
        context_blocks = nn.ModuleList()
        for scale_id, num_latent in enumerate(self.latent_scales): # latent scales from top (smallest) to bottom (largest)
            scale_size = 2 ** (len(self.latent_scales) - scale_id)
            if self.start_scale_at_x:
                scale_size /= 2
            for _ in range(num_latent):
                context_blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=int(scale_size), stride=int(scale_size)),
                    )
                )
                
        return context_blocks
    
    def get_condition(self, z_list, i):
        # before every block which computes [ p(zl|z>l); q(zl|z>l) ], we add context to s_dec features.
        #  This way distirbution becomes p(zl|z>l, context) and q(zl|z>l, context)
        if len(z_list) > 0:
            ctx = self.decoder_blocks[0].ctx_to_x(z_list[0])
            return self.context_blocks[i-1](ctx)
        return None
    
    def z_L_post_proc(self, z_L):
        z_L = self.decoder_blocks[0].ctx_to_x(z_L)
        return z_L

    def final_deterministic(self, z_list, s_dec):
        
        # Likelihood depends on the aggregated latent variables, not s_dec
        up = nn.Upsample(size=s_dec.shape[-1], mode='bilinear')
        const = float(math.sqrt(len(z_list) - 1))
        s_dec = torch.stack([up(z) for z in z_list[1:]], 0).sum(0) / const
        s_dec = self.z_to_features(s_dec)
        out_feature = self.post_process(s_dec)
        
        return out_feature

    def init_dct_normalization(self, loader):
        if self.ctx_type == 'dct':
            if hasattr(loader.dataset, 'dataset'):
                self.decoder_blocks[0].dct_scale.data = loader.dataset.dataset.scale[None, :, :self.ctx_size[1], :self.ctx_size[1]]
            else:
                self.decoder_blocks[0].dct_scale.data = loader.dataset.scale[None, :, :self.ctx_size[1], :self.ctx_size[1]]