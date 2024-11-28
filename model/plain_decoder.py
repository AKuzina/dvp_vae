import math
from typing import Optional, Union
import torch.nn as nn
import torch

from utils.vae_layers import DecoderResBlock

class _Decoder(nn.Module):
    def __init__(self):
        super(_Decoder, self).__init__()

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

    @staticmethod
    def get_num_lik_params(likelihood: str) -> int:
        return {
            'bernoulli': 1,
            'gaussian': 2,
            'gaussian_zero': 1,
            'logistic': 2,
            'logistic_mixture': 3,
            'logistic_mixture_5bit': 3,
            'discretized_gaussian': 2,
        }[likelihood]


class PlainDecoder(_Decoder):
    def __init__(self,
                 z_dim: int,
                 num_ch: int,
                 data_ch: int,
                 data_dim: int,
                 likelihood: str,
                 weight_norm: bool,
                 batch_norm: bool,
                 **kwargs,
                 ):
        super(PlainDecoder, self).__init__()

        self.num_p_param = _Decoder.get_num_lik_params(likelihood)
        self.data_ch = data_ch
        if self.num_p_param == 1:
            act_out = nn.Sigmoid()
        else:
            act_out = nn.Identity()

        conv_param = self.get_conv_params(data_dim)
        n_layers = len(conv_param['strides'])
        channels = num_ch * (2 ** n_layers)
        layers = [
            nn.Upsample(conv_param['upsample'], mode='nearest'),
            nn.Conv2d(z_dim, channels, kernel_size=3, padding=1)
        ]
        for i in range(n_layers+1):
            stride = conv_param['strides'][i] if i < n_layers else 1
            out_ch = int(channels/2) if stride != 1 else channels
            params = {
                'in_channels': channels,
                'hid_channels': channels*6,
                'out_channels': out_ch,
                'stride': stride,
                'activation': nn.SiLU(),
                'weight_norm': weight_norm,
                'batch_norm': batch_norm
            }
            channels = out_ch
            layers.append(
                DecoderResBlock(**params)
            )
        layers += [
            nn.ELU(),
            nn.Conv2d(channels, self.num_p_param*self.data_ch, 3, padding=1),
            act_out
        ]
        self.p_x = nn.Sequential(*layers)
        self.init()

    def forward(self, z):
        z = z.reshape(z.shape[0], -1, 1, 1)
        pxz_param = self.p_x(z)
        return torch.chunk(pxz_param, self.num_p_param, dim=1)

    @staticmethod
    def get_conv_params(data_dim):
        return {
            28: {  # 1 - 7 - 7 - 14 - 14 - 28
                'upsample': 7,
                'strides':  [1, -1, 1, -1],
            },
            32: {  # 1 - 4 - 8 - 8 - 16 - 16 - 32
                'upsample': 4,
                'strides':  [-1, 1, -1, 1, -1],
            },
            64: {  # 1 - 4 - 8 - 8 - 16 - 16 - 32 - 32 - 64
                'upsample': 4,
                'strides':  [-1, 1, -1, 1, -1, 1, -1],
            }
        }[data_dim]
