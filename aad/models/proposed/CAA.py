from typing import Optional, Sequence

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

class CAA(nn.Module):
    def __init__(
            self,
            channels: int,

            v_kernel_size: int = 3,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3,1,1)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.conv1(self.avg_pool(x)))))

        return attn_factor
def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value
def calculate_padding(kernel_size, dilation):
    return (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2


class InceptionBottleneck(BaseModule):
    """Bottleneck with Inception module"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3,3,5,5,7,7),
            dilations: Sequence[int] = (1,2,1,2,1,2),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 7,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.pre_conv = ConvModule(
            in_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        padding1 = (calculate_padding(kernel_sizes[0], dilations[0]), 0)
        padding2 = (calculate_padding(kernel_sizes[1], dilations[1]), 0)
        padding3 = (calculate_padding(kernel_sizes[2], dilations[2]), 0)
        padding4 = (calculate_padding(kernel_sizes[3], dilations[3]), 0)
        padding5 = (calculate_padding(kernel_sizes[4], dilations[4]), 0)
        padding6 = (calculate_padding(kernel_sizes[5], dilations[5]), 0)

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, (kernel_sizes[0],1), 1,
                                  padding1, (dilations[0],1),
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, (kernel_sizes[1],1), 1,
                                   padding2, (dilations[1],1),
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, (kernel_sizes[2],1), 1,
                                  padding3, (dilations[2],1),
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, (kernel_sizes[3],1), 1,
                                  padding4, (dilations[3],1),
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, (kernel_sizes[4],1), 1,
                                  padding5, (dilations[4],1),
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv5 = ConvModule(hidden_channels, hidden_channels, (kernel_sizes[5],1), 1,
                                   padding6, (dilations[5],1),
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)



        if with_caa:
            self.caa_factor = CAA(hidden_channels,caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x
        x = self.dw_conv(x)


        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)+ self.dw_conv5(x)


        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        return x
