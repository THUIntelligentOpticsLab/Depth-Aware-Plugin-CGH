"""
This is the script that is used for the wave propagation using the angular spectrum method (ASM). Refer to
Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005, for principle details.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import math
import torch
import numpy as np
import tools.utils as utils
import torch.fft
import torch.fft as tfft

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
channel = 1
prop_dist = (-20 * cm, -20 * cm, -20 * cm)[channel]
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch

def propagation_ASM(u_in, feature_size=feature_size, wavelength=wavelength, z=prop_dist, linear_conv=False,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32, band_lm=True):
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, pytorch=True, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8),
                                dtype=dtype)
        # get real/img components
        if band_lm:
            H_real, H_imag = utils.polar_to_rect(H_filter, H_exp)
        else:
            H_real, H_imag = utils.polar_to_rect(torch.ones_like(H_exp), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = utils.ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H


    U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')
    H = H.to(U1.device)
    U2 = H * U1
    u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out
