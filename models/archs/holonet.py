"""HoloNet implementation for hologram generation."""
import torch
from torch import nn
from models.backbones.unet import Unet
from tools.propagation.ASM import propagation_ASM


def vector_to_phase(phase_logits: torch.Tensor) -> torch.Tensor:
    phase_real, phase_imag = torch.chunk(phase_logits, 2, dim=1)
    return torch.atan2(phase_imag, phase_real + 1e-8)


def complex_to_channels(field: torch.Tensor, mode: str = 'real_imag') -> torch.Tensor:
    if mode == 'real_imag':
        return torch.cat((field.real, field.imag), dim=1)
    if mode == 'amp_phase':
        return torch.cat((field.abs(), field.angle()), dim=1)
    raise ValueError(f'Unsupported input_mode: {mode}')


class FinalPhaseOnlyUnet(nn.Module):
    def __init__(self,
                 num_down: int = 8,
                 num_features_init: int = 32,
                 max_features: int = 256,
                 norm='group',
                 num_in: int = 2,
                 ft: bool = False):
        super().__init__()
        self.ft = ft
        self.net = Unet(
            in_channels=num_in,
            out_channels=2,
            nf0=num_features_init,
            num_down=num_down,
            max_channels=max_features,
            use_dropout=False,
            upsampling_mode='transpose',
            norm=norm,
            outermost_linear=True,
            ft=self.ft
        )

    def forward(self, amp_phase: torch.Tensor, distance: float = None) -> torch.Tensor:
        out = self.net(amp_phase, distance)
        return vector_to_phase(out)


class HoloNet(nn.Module):
    def __init__(self, ft: bool = False, input_mode: str = 'real_imag'):
        super().__init__()
        self.ft = ft
        self.input_mode = input_mode
        self.final_phase_only = FinalPhaseOnlyUnet(4, 32, ft=self.ft, num_in=2)

    def forward(self, target_amps, Hbackward=None, distance=None):
        if distance is not None:
            distance_H = distance
        else:
            distance_H = None

        slmfield = propagation_ASM(target_amps, precomped_H=Hbackward)

        slm_features = complex_to_channels(slmfield, mode=self.input_mode)
        holophase = self.final_phase_only(slm_features, distance=distance_H)

        return slmfield, holophase
