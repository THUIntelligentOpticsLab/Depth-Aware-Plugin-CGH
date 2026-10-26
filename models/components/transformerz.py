import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np


class DPConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class OTFRouting(nn.Module):
    """OTF-phase encoder from the manuscript's network structure figure.

    Ten stride-2 DPConv blocks map the existing 1072 x 1920 OTF to
    32 x 2 x 2, followed by GAP, FC(32, 64), FC(64, 4), and softmax.
    Explicit stem_channels and old checkpoints retain their original encoder.
    """
    def __init__(self, in_ch=1, num_experts=4, hidden_dim=64,
                 stem_channels=(16, 32, 32, 32, 32, 32, 32, 32, 32, 32)):
        super().__init__()
        self.num_experts = num_experts

        self.encoder = self._make_encoder(in_ch, stem_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(stem_channels[-1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _make_encoder(in_ch, stem_channels):
        layers = []
        prev_ch = in_ch
        for ch in stem_channels:
            block = DPConv(prev_ch, ch, stride=2)
            # Preserve signal scale through the deeper encoder. Only the
            # pointwise convolution is followed by ReLU in each DPConv.
            nn.init.kaiming_normal_(block.block[0].weight, nonlinearity='linear')
            nn.init.kaiming_normal_(block.block[1].weight, nonlinearity='relu')
            layers.append(block)
            prev_ch = ch
        return nn.Sequential(*layers)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Existing runs used four DPConv blocks (8, 16, 32, 32). Rebuild
        # that encoder before recursive loading so strict resume/evaluation
        # preserves its weights, outputs, and optimizer parameter ordering.
        channels = []
        while prefix + f'encoder.{len(channels)}.block.1.weight' in state_dict:
            channels.append(state_dict[prefix + f'encoder.{len(channels)}.block.1.weight'].shape[0])
        current_channels = [block.block[1].out_channels for block in self.encoder]
        if channels and channels != current_channels:
            in_ch = state_dict[prefix + 'encoder.0.block.0.weight'].shape[0]
            self.encoder = self._make_encoder(in_ch, channels).to(self.fc1.weight)
            self.encoder.train(self.training)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, H_phase):
        x = self.encoder(H_phase)
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return self.softmax(x)

class TransformerRoutingWithZ(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, num_experts=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32//4 * 32//4 + 1, embed_dim))  # +1 for z token
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.z_proj = nn.Linear(1, embed_dim)  # 将标量 z 编码成 token 向量

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_experts),
            nn.Softmax(dim=1)
        )

        self.enable_save = False
        self.save_dir = None
        self.name = None

    def forward(self, x, z):  # x: [B, 1, H, W], z: [B, 1]
        if isinstance(z, float) or isinstance(z, int):
            z = torch.tensor([[z]], dtype=torch.float32, device=x.device)
        elif isinstance(z, torch.Tensor):
            if z.dim() == 1:
                z = z.unsqueeze(1)
            z = z.to(dtype=torch.float32, device=x.device)
        else:
            raise TypeError("z must be a float or torch.Tensor")
        x = self.patch_embed(x)  # [B, C, H/4, W/4]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # -> [B, N, C]

        z_token = self.z_proj(z).unsqueeze(1)  # [B, 1, C]
        x = torch.cat([z_token, x], dim=1)     # prepend z token: [B, N+1, C]

        x = x + self.pos_embedding[:, :x.size(1)]  # 加 positional encoding
        x = self.transformer(x)  # [B, N+1, C]

        z_output = x[:, 0]  # 取出第一个 token（z token 的输出）

        if self.enable_save and self.save_dir:
            x_np = z_output.detach().cpu().numpy()
            np.save(f"{self.save_dir}/z_output_{self.name}.npy", x_np)
            # 自动关闭（只保存一次）
            self.enable_save = False

        return self.mlp_head(z_output)  # 输出融合权重

class TransformerRouting(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, num_experts=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32//4 * 32//4, embed_dim))  # 4x downsample
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # x: [B, 1, H, W]
        x = self.patch_embed(x)  # [B, embed_dim, H/4, W/4]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # flatten to sequence
        x = x + self.pos_embedding[:, :H*W]
        x = self.transformer(x)  # [B, N, C]
        x = x.mean(dim=1)  # Global average over tokens
        return self.mlp_head(x)

