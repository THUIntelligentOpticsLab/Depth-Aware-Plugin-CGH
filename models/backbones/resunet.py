import torch
import torch.nn as nn
import math
from models.components.ftmodule import FT_module
import numpy as np


def _group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _make_norm(kind, num_channels: int) -> nn.Module:
    if kind in (None, 'none'):
        return nn.Identity()
    if kind == 'batch':
        return nn.BatchNorm2d(num_channels)
    if kind == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=True)
    if kind == 'group':
        return nn.GroupNorm(_group_count(num_channels), num_channels)
    if callable(kind):
        try:
            return kind(num_channels, affine=True)
        except TypeError:
            return kind(num_channels)
    raise ValueError(f'Unsupported norm kind: {kind}')


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, ft=False, norm='group'):
        super(ResidualConv, self).__init__()

        self.ft = ft
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            _make_norm(norm, output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            _make_norm(norm, output_dim),
        )
        if stride == 1:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
                _make_norm(norm, output_dim),
            )
        elif stride == 2:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
                _make_norm(norm, output_dim),
            )
        self.rl = nn.ReLU()

        if self.ft:
            self.out_ft = FT_module(output_dim)

    def forward(self, x, distance=None):
        if self.ft and distance is not None:
            return self.out_ft(self.rl(self.conv_block(x) + self.conv_skip(x)), distance=distance)
        else:
            return self.rl(self.conv_block(x) + self.conv_skip(x))


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride, ft=False, norm='group'):
        super(Upsample, self).__init__()

        self.ft = ft
        self.upsample = [nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )]
        self.upsample.append(_make_norm(norm, output_dim))
        self.upsample.append(nn.ReLU())
        self.upsample = nn.Sequential(*self.upsample)

        if self.ft:
            self.out_ft = FT_module(output_dim)

    def forward(self, x, distance=None):
        if self.ft and distance is not None:
            return self.out_ft(self.upsample(x), distance=distance)
        else:
            return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, in_channel, out_channel, filters=[8, 16, 32, 64], ft=False, norm='group'):
        super(ResUnet, self).__init__()

        self.ft = ft
        self.input_layer = ResidualConv(in_channel, filters[0], 1, 1, ft=self.ft, norm=norm)

        self.residual_down_21 = ResidualConv(filters[0], filters[1], 2, 1, ft=self.ft, norm=norm)
        self.residual_down_31 = ResidualConv(filters[1], filters[2], 2, 1, ft=self.ft, norm=norm)
        self.residual_down_41 = ResidualConv(filters[2], filters[3], 2, 1, ft=self.ft, norm=norm)

        self.upsample_32 = Upsample(filters[3], filters[2], 2, 2, ft=self.ft, norm=norm)
        self.residual_conv32 = ResidualConv(2*filters[2], filters[2], 1, 1, ft=self.ft, norm=norm)

        self.upsample_22 = Upsample(filters[2], filters[1], 2, 2, ft=self.ft, norm=norm)
        self.residual_conv22 = ResidualConv(2*filters[1], filters[1], 1, 1, ft=self.ft, norm=norm)
        self.upsample_23 = Upsample(filters[2], filters[1], 2, 2, ft=self.ft, norm=norm)
        self.residual_conv23 = ResidualConv(3*filters[1], filters[1], 1, 1, ft=self.ft, norm=norm)

        self.upsample_12 = Upsample(filters[1], filters[0], 2, 2, ft=self.ft, norm=norm)
        self.residual_conv12 = ResidualConv(2*filters[0], filters[0], 1, 1, ft=self.ft, norm=norm)
        self.upsample_13 = Upsample(filters[1], filters[0], 2, 2, ft=self.ft, norm=norm)
        self.residual_conv13 = ResidualConv(3*filters[0], filters[0], 1, 1, ft=self.ft, norm=norm)
        self.upsample_14 = Upsample(filters[1], filters[0], 2, 2, ft=self.ft, norm=norm)
        self.residual_conv14 = ResidualConv(4*filters[0], filters[0], 1, 1, ft=self.ft, norm=norm)

        self.output_layer1 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
        )
        self.output_layer3 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
        )
        self.output_layer4 = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
        )

        self.enable_save = False
        self.save_dir = None
        self.name = None

    def forward(self, x, distance=None):
        # Encode
        x1_1 = self.input_layer(x, distance=distance)
        x2_1 = self.residual_down_21(x1_1, distance=distance)
        x3_1 = self.residual_down_31(x2_1, distance=distance)
        x4_1 = self.residual_down_41(x3_1, distance=distance)
        # Decode
        x3_2 = self.upsample_32(x4_1, distance=distance)
        x3_2 = torch.cat([x3_1, x3_2], dim=1)
        x3_2 = self.residual_conv32(x3_2, distance=distance)

        x2_2 = self.upsample_22(x3_1, distance=distance)
        x2_2 = torch.cat([x2_1, x2_2], dim=1)
        x2_2 = self.residual_conv22(x2_2, distance=distance)
        x2_3 = self.upsample_23(x3_2, distance=distance)
        x2_3 = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x2_3 = self.residual_conv23(x2_3, distance=distance)

        x1_2 = self.upsample_12(x2_1, distance=distance)
        x1_2 = torch.cat([x1_1, x1_2], dim=1)
        x1_2 = self.residual_conv12(x1_2, distance=distance)
        x1_3 = self.upsample_13(x2_2, distance=distance)
        x1_3 = torch.cat([x1_1, x1_2, x1_3], dim=1)
        x1_3 = self.residual_conv13(x1_3, distance=distance)
        x1_4 = self.upsample_14(x2_3, distance=distance)
        x1_4 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1_4 = self.residual_conv14(x1_4, distance=distance)

        output1_4 = self.output_layer4(x1_4)

        if self.enable_save and self.save_dir:
            x1_1_np = x1_1.detach().cpu().numpy()
            np.save(f"{self.save_dir}/x1_1_{self.name}.npy", x1_1_np)
            x1_2_np = x1_2.detach().cpu().numpy()
            np.save(f"{self.save_dir}/x1_2_{self.name}.npy", x1_2_np)
            x1_3_np = x1_3.detach().cpu().numpy()
            np.save(f"{self.save_dir}/x1_3_{self.name}.npy", x1_3_np)
            x1_4_np = x1_4.detach().cpu().numpy()
            np.save(f"{self.save_dir}/x1_4_{self.name}.npy", x1_4_np)
            # 自动关闭（只保存一次）
            self.enable_save = False

        return output1_4
