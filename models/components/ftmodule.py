import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.transformerz import OTFRouting


class FT_module(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enable_save = False
        self.save_dir = None
        self.name = None

        self.adapt = FT_conv(
            channels_in=channels,
            channels_out=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            num_experts=4
        )

    def _save_debug(self, x: torch.Tensor, adapted: torch.Tensor, out: torch.Tensor):
        if not (self.enable_save and self.save_dir):
            return

        os.makedirs(self.save_dir, exist_ok=True)
        suffix = self.name if self.name is not None else "unnamed"

        np.save(
            os.path.join(self.save_dir, f"x_{suffix}.npy"),
            x.detach().cpu().numpy()
        )
        np.save(
            os.path.join(self.save_dir, f"adapted_{suffix}.npy"),
            adapted.detach().cpu().numpy()
        )
        np.save(
            os.path.join(self.save_dir, f"out_{suffix}.npy"),
            out.detach().cpu().numpy()
        )

        # 只保存一次
        self.enable_save = False

    def forward(self, x, distance):
        adapted = self.adapt(x, distance)

        # Follow the manuscript's network structure figure: residual feature
        # modulation, without a spatial mask. The text abbreviates this as
        # F' = W(z) * F, whereas the figure explicitly includes the skip path.
        out = x + adapted

        self._save_debug(x, adapted, out)
        return out


class FT_conv(nn.Module):
    """
    按样本进行动态卷积：
    即：每个样本根据对应的 OTF phase 生成一组 routing weights，
    再融合 preset kernels 得到一个卷积核。

    注意：
    - 当前实现仅支持 batch size = 1
    - distance 期望为二元组: (distance_H, distance_num)
        distance_H: complex tensor, shape [1, 1, H, W]
        distance_num: scalar tensor / float-like, 当前仅保留接口，不参与 routing
    """
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        num_experts: int = 4
    ):
        super().__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.use_bias = bias

        self.enable_save = False
        self.save_dir = None
        self.name = None

        self.routing = OTFRouting(
            in_ch=1,
            num_experts=num_experts,
            hidden_dim=64
        )

        # [E, Cout, Cin, K, K]
        weight_pool = torch.empty(
            num_experts, channels_out, channels_in, kernel_size, kernel_size
        )
        for i in range(num_experts):
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(weight_pool)

        if self.use_bias:
            self.bias_pool = nn.Parameter(torch.empty(num_experts, channels_out))
            fan_in = channels_in * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)
        else:
            self.register_parameter("bias_pool", None)

        self.last_fused_weight = None

    def _check_inputs(self, x: torch.Tensor, distance):
        if not isinstance(distance, (tuple, list)) or len(distance) != 2:
            raise ValueError(
                "distance must be a tuple/list of length 2: (distance_H, distance_num)"
            )

        distance_H, distance_num = distance

        if x.dim() != 4:
            raise ValueError(f"x must be 4D [B,C,H,W], got shape {tuple(x.shape)}")

        if x.size(0) != 1:
            raise ValueError(
                f"Current FT_conv only supports batch size = 1, got x.shape={tuple(x.shape)}"
            )

        if not torch.is_tensor(distance_H):
            raise TypeError("distance_H must be a torch.Tensor")

        if distance_H.dim() != 4:
            raise ValueError(
                f"distance_H must be 4D [B,1,H,W], got shape {tuple(distance_H.shape)}"
            )

        if distance_H.size(0) != 1:
            raise ValueError(
                f"Current FT_conv only supports distance_H batch size = 1, got shape={tuple(distance_H.shape)}"
            )

        if distance_H.size(1) != 1:
            raise ValueError(
                f"distance_H must have 1 channel, got shape={tuple(distance_H.shape)}"
            )

        if not torch.is_complex(distance_H):
            raise TypeError(
                f"distance_H must be a complex tensor, got dtype={distance_H.dtype}"
            )

        return distance_H, distance_num

    def _save_routing_weights(self, routing_weights: torch.Tensor):
        if not (self.enable_save and self.save_dir):
            return

        os.makedirs(self.save_dir, exist_ok=True)
        suffix = self.name if self.name is not None else "unnamed"

        np.save(
            os.path.join(self.save_dir, f"routing_weights_{suffix}.npy"),
            routing_weights.detach().cpu().numpy()
        )

        # 只保存一次
        self.enable_save = False

    def forward(self, x, distance):
        distance_H, distance_num = self._check_inputs(x, distance)

        # 保留接口兼容性；当前 routing 仅使用 arg(H)
        _ = distance_num

        # H_phase: [1,1,H,W]
        H_phase = distance_H.angle()

        # routing_out: [1, E]
        routing_out = self.routing(H_phase)

        if routing_out.dim() != 2 or routing_out.shape[0] != 1 or routing_out.shape[1] != self.num_experts:
            raise ValueError(
                f"routing output must have shape [1, {self.num_experts}], got {tuple(routing_out.shape)}"
            )

        # [E]
        routing_vec = routing_out[0]

        self._save_routing_weights(routing_vec)

        # fuse experts:
        # weight_pool: [E, Cout, Cin, K, K]
        # routing_vec: [E]
        fused_weight = (self.weight_pool * routing_vec.view(self.num_experts, 1, 1, 1, 1)).sum(dim=0)
        # fused_weight: [Cout, Cin, K, K]

        self.last_fused_weight = fused_weight.detach().cpu()

        if self.use_bias:
            # bias_pool: [E, Cout]
            fused_bias = (self.bias_pool * routing_vec.view(self.num_experts, 1)).sum(dim=0)
        else:
            fused_bias = None

        out = F.conv2d(
            x,
            fused_weight,
            fused_bias,
            stride=self.stride,
            padding=self.padding
        )
        return out


if __name__ == "__main__":
    # 最小可运行测试
    x = torch.randn(1, 32, 64, 64)

    # 构造一个复数形式的 distance_H
    real = torch.randn(1, 1, 64, 64)
    imag = torch.randn(1, 1, 64, 64)
    distance_H = torch.complex(real, imag)

    distance_num = torch.tensor([0.2])  # 例如 0.2 m
    distance = (distance_H, distance_num)

    model = FT_module(32)
    out = model(x, distance)

    print("input shape :", x.shape)
    print("output shape:", out.shape)
