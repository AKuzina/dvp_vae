import torch.nn as nn
import torch

from utils.vae_layers import EncoderResBlock


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()

    def init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }[name]


class PlainEncoder(_Encoder):
    def __init__(self,
                 z_dim: int,
                 num_ch: int,
                 data_ch: int,
                 data_dim: int,
                 weight_norm: bool,
                 batch_norm: bool,
                 **kwargs,
                 ):
        super(PlainEncoder, self).__init__()
        self.num_q_param = 2
        self.z_dim = z_dim
        conv_param = self.get_conv_params(data_dim)
        n_layers = len(conv_param['strides'])
        layers = [nn.Conv2d(data_ch, num_ch, kernel_size=1)]
        channels = num_ch
        for i in range(n_layers):
            params = {
                'in_channels': channels,
                'hid_channels': channels,
                'out_channels': channels * 2,
                'stride': conv_param['strides'][i],
                'activation': None if i+1 == n_layers else nn.SiLU(),
                'weight_norm': weight_norm,
                'batch_norm': batch_norm,
                'num_blocks': 1
            }
            channels *= 2
            layers.append(
                EncoderResBlock(**params)
            )

        layers.append(nn.Conv2d(channels, 2*z_dim, kernel_size=1, padding=0))
        layers.append(nn.Flatten())
        self.q_z = nn.Sequential(*layers)
        self.init()

    def forward(self, x) -> tuple:
        q_param = self.q_z(x)
        return torch.chunk(q_param, self.num_q_param, dim=1)

    @staticmethod
    def get_conv_params(data_dim: int) -> dict:
        return {
            28: {
                'strides': [2]*5,
            },
            32: {
                'strides': [2]*5,
            },
            64: {
                'strides': [2]*6,
            }
        }[data_dim]


class LadderEncoder(_Encoder):
    def __init__(self,
                 num_ch: int,
                 scale_ch_mult: float,
                 block_ch_mult: float,
                 data_ch: int,
                 num_init_blocks: int,
                 data_dim: int,
                 weight_norm: bool,
                 batch_norm: bool,
                 latent_scales: list,
                 latent_width: list,
                 num_blocks_per_scale: int,
                 activation: str,
                 dset: str = 'mnist',
                 start_scale_at_x: bool = False,
                 ):
        super(LadderEncoder, self).__init__()
        self.dset = dset
        self.conv_block_params = {
            'batch_norm': batch_norm,
            'weight_norm': weight_norm,
            'activation': self.get_activation(activation),
            'use_res': True
        }
        self.image_size = [data_ch, data_dim, data_dim]
        self.block_ch_mult = block_ch_mult
        assert len(latent_width) == len(latent_scales)

        self.num_ch = [num_ch]
        self.start_scale_at_x = start_scale_at_x
        for i in range(len(latent_scales)-1):
            self.num_ch += [int(self.num_ch[-1] * scale_ch_mult)]

        print('Encoder channels', self.num_ch)
        self.pre_process = self.init_pre_process(num_init_blocks)
        self.latent_scales = latent_scales
        self.latent_width = latent_width
        self.encoder_blocks = self.init_encoder_blocks(num_blocks_per_scale)
        self.init()

    def init_pre_process(self, num_init_blocks: int) -> nn.Module:
        pre_process = [nn.Conv2d(self.image_size[0], self.num_ch[0], kernel_size=1)]
        for i in range(num_init_blocks):
            pre_process += [
                EncoderResBlock(self.num_ch[0],
                                int(self.num_ch[0] * self.block_ch_mult),
                                self.num_ch[0],
                                stride=1,
                                num_blocks=2,
                                **self.conv_block_params)
            ]
        return nn.Sequential(*pre_process)

    def init_encoder_blocks(self, num_blocks_per_scale: int) -> tuple:
        backbone = nn.ModuleList()
        pool_k = 2
        if self.start_scale_at_x: # if first scale of latent == data dim
            pool_k = 1
        z_size = self.image_size[-1]
        for s_num, (s, w) in enumerate(zip(self.latent_scales, self.latent_width)):
            for latent in range(s):
                curr_net = []
                for i in range(num_blocks_per_scale):
                    in_ch, out_ch = self.num_ch[s_num], self.num_ch[s_num]
                    if s_num > 0 and latent == 0 and i == 0:
                        in_ch = self.num_ch[s_num-1]
                    curr_net += [
                        EncoderResBlock(
                            in_ch,
                            int(in_ch * self.block_ch_mult),
                            out_ch,
                            stride=1,
                            num_blocks=2,
                            **self.conv_block_params)
                    ]
                if latent == 0:
                    z_size /= pool_k
                    print(f'Scale {s_num+1}, {s} latents, out shape: {int(z_size)}')
                    # if this is the first latent of the scale -> reduce size
                    curr_net += [nn.AvgPool2d(kernel_size=pool_k, stride=pool_k)]
                    pool_k = 1
                backbone.append(nn.Sequential(*curr_net))
            pool_k *= 2
        return backbone

    def forward(self, x) -> list:
        x = prep_x(self.dset, x)
        d = self.pre_process(x)
        s_enc = []
        for enc in self.encoder_blocks:
            d = enc(d)
            s_enc.append(d)
        return s_enc


class SmallLadderEncoder(LadderEncoder):
    def __init__(self,
                 num_ch: int,
                 scale_ch_mult: float,
                 block_ch_mult: float,
                 data_ch: int,
                 num_init_blocks: int,
                 data_dim: int,
                 weight_norm: bool,
                 batch_norm: bool,
                 latent_scales: list,
                 latent_width: list,
                 num_blocks_per_scale: int,
                 activation: str,
                 dset: str = 'mnist',
                 start_scale_at_x: bool = False,
                 ):
        super(SmallLadderEncoder, self).__init__(
                    num_ch=num_ch,
                    scale_ch_mult=scale_ch_mult,
                    block_ch_mult=block_ch_mult,
                    data_ch=data_ch,
                    num_init_blocks=num_init_blocks,
                    data_dim=data_dim,
                    weight_norm=weight_norm,
                    batch_norm=batch_norm,
                    latent_scales=latent_scales,
                    latent_width=latent_width,
                    num_blocks_per_scale=num_blocks_per_scale,
                    activation=activation,
                    dset=dset,
                    start_scale_at_x=start_scale_at_x,
        )

    def init_encoder_blocks(self, num_blocks_per_scale: int) -> tuple:
        backbone = nn.ModuleList()
        pool_k = 2
        if self.start_scale_at_x:  # if first scale of latent == data dim
            pool_k = 1
        z_size = self.image_size[-1]
        for s_num, (s, w) in enumerate(zip(self.latent_scales, self.latent_width)):
            # for latent in range(s):
            if s > 0:
                curr_net = []
                for i in range(num_blocks_per_scale):
                    in_ch, out_ch = self.num_ch[s_num], self.num_ch[s_num]
                    if s_num > 0 and i == 0:
                        in_ch = self.num_ch[s_num - 1]
                    curr_net += [
                        EncoderResBlock(
                            in_ch,
                            int(in_ch * self.block_ch_mult),
                            out_ch,
                            stride=1,
                            num_blocks=2,
                            **self.conv_block_params)
                    ]

                z_size /= pool_k
                print(f'Scale {s_num + 1}, {s} latents, out shape: {int(z_size)}')
                # if this is the first latent of the scale -> reduce size
                curr_net += [nn.AvgPool2d(kernel_size=pool_k, stride=pool_k)]
                pool_k = 1
                backbone.append(nn.Sequential(*curr_net))
            pool_k *= 2
        return backbone

    def forward(self, x) -> list:
        x = prep_x(self.dset, x)
        d = self.pre_process(x)
        s_enc = []
        i = 0
        for enc in self.encoder_blocks:
            while self.latent_scales[i] == 0:
                i += 1
            num_latents = self.latent_scales[i]
            d = enc(d)
            for _ in range(num_latents):
                s_enc.append(d)
            i += 1
        return s_enc


def prep_x(dset, x):
    z = x * 127.5 + 127.5
    if dset == 'cifar10':
        z = (z - 120.63838) / 64.16736
    else:
        z = (z - 127.5) / 127.5
    return z
