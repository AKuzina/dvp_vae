import torch.nn as nn

from utils.nn import _ResBlock, ConvBlock


class EncoderResBlock(_ResBlock):
    def __init__(self, in_channels, hid_channels, out_channels, activation,
                 weight_norm, batch_norm, stride=1, num_blocks=2, use_res=True, padding_mode='zeros'):
        super(EncoderResBlock, self).__init__(in_channels, out_channels, stride, use_res)
        conv_params = {
            'activation': activation,
            'weight_norm': weight_norm,
            'batch_norm': batch_norm,
            'forward': True,
            'padding_mode': 'zeros',
        }

        h_blocks = [
            ConvBlock(in_ch=in_channels, out_ch=hid_channels, kernel_size=1, **conv_params)
        ]
        for i in range(num_blocks):
            h_blocks.append(
                ConvBlock(
                    in_ch=hid_channels,
                    out_ch=hid_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride if i + 1 == num_blocks else 1,
                    **conv_params
            ))
        h_blocks.append(
            ConvBlock(in_ch=hid_channels, out_ch=out_channels, kernel_size=1,
                      **conv_params)
        )
        self.net = nn.Sequential(*h_blocks)


class DecoderResBlock(_ResBlock):
    def __init__(self, in_channels, hid_channels, out_channels, activation,
                 weight_norm, batch_norm, stride=1, mode='1x5', use_res=True, zero_last=False, padding_mode='zeros'):
        super(DecoderResBlock, self).__init__(in_channels, out_channels, stride, use_res)
        assert mode in ['1x5', '2x3']
        h_blocks = []

        if stride == -1:
            h_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if batch_norm:
            h_blocks.append(nn.BatchNorm2d(in_channels, momentum=0.05))
        params = {
            'batch_norm': batch_norm,
            'weight_norm': weight_norm,
            'activation': activation,
            'padding_mode': padding_mode,
        }
        h_blocks += [
            ConvBlock(in_channels, hid_channels, 1, forward=False, **params)
        ]
        if mode == '1x5':
            h_blocks += [
                ConvBlock(hid_channels, hid_channels, 5, forward=False, padding=2,
                          groups=hid_channels, stride=stride,  **params)
                         ]
        elif mode == '2x3':
            h_blocks += [
                ConvBlock(hid_channels, hid_channels, 3, forward=False, padding=1,
                          stride=stride, **params),
                ConvBlock(hid_channels, hid_channels, 3, forward=False, padding=1,
                          stride=1, **params),
            ]
        h_blocks += [
            nn.Conv2d(hid_channels, out_channels, 1)
        ]
        self.last_conv_id = len(h_blocks) - 1

        if batch_norm:
            h_blocks.append(nn.BatchNorm2d(out_channels, momentum=0.05))
        self.net = nn.Sequential(*h_blocks)

        if zero_last:
            nn.init.zeros_(self.net[self.last_conv_id].weight)
            nn.init.zeros_(self.net[self.last_conv_id].bias)

