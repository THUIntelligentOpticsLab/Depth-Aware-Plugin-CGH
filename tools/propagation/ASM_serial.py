import torch
import torch.nn as nn
import tools.utils as utils
import torch.fft as tfft
import math
import numpy as np
from tools.propagation.ASM import propagation_ASM

class SerialProp(nn.Module):
    def __init__(self, cpx, wavelength, feature_size, prop_dists_from_wrp=None, linear_conv=True):
        super(SerialProp, self).__init__()


        self.prop_H_ls = []
        if prop_dists_from_wrp is not None:
            for dist in prop_dists_from_wrp:
                H = propagation_ASM(
                    cpx,
                    feature_size=feature_size,
                    wavelength=wavelength,
                    z=dist,
                    linear_conv=linear_conv,
                    return_H=True
                )
                self.prop_H_ls.append(H)
        self.prop_H = torch.concatenate(self.prop_H_ls, dim=1)

    def forward(self, u_in):
        if u_in.dtype == torch.float32:
            u_in = torch.exp(1j * u_in)

        u_out = propagation_ASM(u_in, precomped_H=self.prop_H)
        return u_out
