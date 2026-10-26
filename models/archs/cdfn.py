import torch
from torch import nn
from models.backbones.resunet import ResUnet
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


class InitialPhaseUnet(nn.Module):
    """Initial phase estimation network.

    Computes the initial input phase given a target amplitude.
    """
    def __init__(self, ft: bool = False):
        """
        Initialize network.

        Args:
            ft: Whether to use feature transform modules
        """
        super().__init__()
        self.ft = ft
        self.net = ResUnet(
            in_channel=1,
            out_channel=2,
            filters=[32, 32, 128, 128],
            ft=self.ft,
            norm='group'
        )

    def forward(self, amp: torch.Tensor, distance: float = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            amp: Input amplitude
            distance: Optional propagation distance for FT modules

        Returns:
            Estimated phase
        """
        out = self.net(amp, distance=distance)
        return vector_to_phase(out)


class FinalPhaseOnlyUnet(nn.Module):
    """Final phase refinement network.

    Computes the final SLM phase given amplitude and initial phase.
    """
    def __init__(self, ft: bool = False):
        """
        Initialize network.

        Args:
            ft: Whether to use feature transform modules
        """
        super().__init__()
        self.ft = ft
        self.net = ResUnet(
            in_channel=2,
            out_channel=2,
            filters=[32, 32, 128, 128],
            ft=self.ft,
            norm='group'
        )

    def forward(self, amp_phase: torch.Tensor, distance: float = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            amp_phase: Concatenated amplitude and phase
            distance: Optional propagation distance for FT modules

        Returns:
            Refined phase
        """
        out = self.net(amp_phase, distance=distance)
        return vector_to_phase(out)


class CDFN(nn.Module):
    def __init__(self, ft=False, input_mode: str = 'real_imag'):
        super().__init__()

        self.ft = ft
        self.input_mode = input_mode
        self.final_phase_only = FinalPhaseOnlyUnet(ft=self.ft)

    def forward(self, target_amps, Hbackward=None, distance=None):
        if distance is not None:
            distance_H = distance
        else:
            distance_H = None

        slmfield = propagation_ASM(target_amps, precomped_H=Hbackward)

        slm_features = complex_to_channels(slmfield, mode=self.input_mode)
        holophase = self.final_phase_only(slm_features, distance=distance_H)

        return slmfield, holophase
