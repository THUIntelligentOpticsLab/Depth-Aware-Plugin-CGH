"""
Test script for holographic reconstruction distance sweep.
Load a trained checkpoint, predict hologram phase, and reconstruct from
20cm to 30cm with 1cm interval.

Save structure:
save_dir/
├── poh/
│   ├── 20cm/
│   ├── 21cm/
│   ├── ...
│   ├── 30cm/
│   └── all/
├── recon/
│   ├── 20cm/
│   ├── 21cm/
│   ├── ...
│   ├── 30cm/
│   └── all/
└── target/
    └── all/

Usage example:
python test_2d_diffdn.py \
    --channel 1 \
    --valid_root ./data/DIV2K_valid_HR \
    --ckpt_path /path/to/your.ckpt \
    --save_dir ./test_results_sweep \
    --batch_size 1
"""

import os
import math
import argparse
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import imageio.v3 as iio
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

import tools.utils as utils
from tools.propagation.ASM import propagation_ASM
from models.archs.cdfn import CDFN
from models.archs.holonet import HoloNet
from datasets.rgbd import RGBD


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hologram reconstruction sweep test")
    p.add_argument('--model', type=str, default=None, choices=['cdfn', 'holonet'],
                   help='Override model architecture used at training time')
    p.add_argument('--input_mode', type=str, default=None, choices=['real_imag', 'amp_phase'],
                   help='Override feature representation for real-valued models')
    p.add_argument('--channel', type=int, default=1,
                   help='Channel selection: red:0, green:1, blue:2, rgb:3')
    p.add_argument('--valid_root', type=str, default='./data/DIV2K_valid_HR',
                   help='validation/test data root')
    p.add_argument('--batch_size', type=int, default=1,
                   help='test batch size, recommended 1')
    p.add_argument('--scale_output', type=float, default=0.95,
                   help='Scale of output applied to reconstructed intensity from SLM')
    p.add_argument('--ckpt_path', type=str, required=True,
                   help='checkpoint path')
    p.add_argument('--save_dir', type=str, default='./test_results_sweep',
                   help='directory to save results')
    p.add_argument('--ft', type=utils.str2bool, default=None,
                   help='Override FT flag from training config')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--z_start_cm', type=int, default=20)
    p.add_argument('--z_end_cm', type=int, default=30)
    p.add_argument('--z_step_cm', type=int, default=1)
    return p.parse_args()


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor image to uint8 numpy array in [0, 255].
    Input shape can be:
      - (1, H, W)
      - (H, W)
    """
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    img = img.detach().float().cpu()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255.0).clamp(0, 255).byte().numpy()
    return img


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prepare_save_dirs(root: str, z_list_cm: List[int]) -> None:
    """
    Create directory tree:
    save_dir/
    ├── poh/{20cm,...,30cm,all}
    ├── recon/{20cm,...,30cm,all}
    └── target/all
    """
    ensure_dir(root)

    poh_root = os.path.join(root, "poh")
    recon_root = os.path.join(root, "recon")
    target_root = os.path.join(root, "target")

    ensure_dir(poh_root)
    ensure_dir(recon_root)
    ensure_dir(target_root)

    ensure_dir(os.path.join(poh_root, "all"))
    ensure_dir(os.path.join(recon_root, "all"))
    ensure_dir(os.path.join(target_root, "all"))

    for z_cm in z_list_cm:
        ensure_dir(os.path.join(poh_root, f"{z_cm:02d}cm"))
        ensure_dir(os.path.join(recon_root, f"{z_cm:02d}cm"))


class HoloSystem(LightningModule):
    """
    Same core logic as training/validation:
    - predict holo_phase from target_amp
    - build holo_cpx = exp(1j * holo_phase)
    - propagate to reconstruction plane
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(ignore=['args'])
        self.args = args
        self._init_system_params(args)
        self._init_model_components()
        self._init_dataset()

    def _init_system_params(self, args: argparse.Namespace) -> None:
        self.batch_size = args.batch_size
        self.channel = args.channel
        self.ft = args.ft

        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        self.cm = cm
        self.mm = mm
        self.um = um
        self.nm = nm

        self.prop_dist = [-200 * mm, -200 * mm, -200 * mm][self.channel]
        self.wavelength = (638 * nm, 520 * nm, 450 * nm)[self.channel]
        self.feature_size = (6.4 * um, 6.4 * um)
        self.pitch = 6.4 * um
        self.pad = False
        self.scale_output = args.scale_output

        self.homography_res = (1072, 1920)
        self.roi_res = (880, 1600)

    def _init_model_components(self) -> None:
        if self.args.model == 'cdfn':
            self.phase_generator = CDFN(ft=self.ft, input_mode=self.args.input_mode)
        elif self.args.model == 'holonet':
            self.phase_generator = HoloNet(ft=self.ft, input_mode=self.args.input_mode)
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

    def _init_dataset(self) -> None:
        self.val_set = RGBD(
            self.args.valid_root,
            channel=self.channel,
            split='val',
            image_res=self.roi_res,
            homography_res=self.homography_res
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def build_backward_H(self, batch_size: int, z: float, device: torch.device) -> torch.Tensor:
        n, m = self.homography_res
        Hbackward_H = propagation_ASM(
            torch.empty(batch_size, 1, n, m, device=device),
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength,
            z=-z,
            linear_conv=self.pad,
            return_H=True
        ).to(self.device)
        return Hbackward_H

    def build_inner_H(self, batch_size: int, z: float, device: torch.device) -> torch.Tensor:
        n, m = self.homography_res
        H = propagation_ASM(
            torch.empty(batch_size, 1, n, m, device=device),
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength,
            z=z,
            linear_conv=self.pad,
            return_H=True,
            band_lm=False
        ).to(self.device)
        return H

    def build_forward_H(self, batch_size: int, z: float, device: torch.device) -> torch.Tensor:
        n, m = self.homography_res
        Hforward_H = propagation_ASM(
            torch.empty(batch_size, 1, n, m, device=device),
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength,
            z=z,
            linear_conv=self.pad,
            return_H=True
        ).to(self.device)
        return Hforward_H

    @torch.no_grad()
    def infer_phase_and_recon_sweep(self, target_amp: torch.Tensor, z_list_m: List[float]):
        device = target_amp.device
        batch_size = target_amp.shape[0]
        target_res = self.roi_res

        all_results = []

        for z in z_list_m:
            Hbackward_H = self.build_backward_H(batch_size, z, device)
            H_inner = self.build_inner_H(batch_size, z, device)
            Hforward_H = self.build_forward_H(batch_size, z, device)

            U_new = target_amp * torch.exp(1j * torch.zeros_like(target_amp))

            slmfield, holo_phase = self.phase_generator(
                U_new,
                Hbackward_H,
                distance=(H_inner, z)
            )

            holo_cpx = torch.exp(1j * holo_phase)
            recon_complex = propagation_ASM(holo_cpx, precomped_H=Hforward_H)
            recon_amp = recon_complex.abs() * math.sqrt(self.scale_output)

            recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)

            all_results.append({
                'z_m': z,
                'holo_phase': holo_phase,
                'recon_amp': recon_amp,
                'recon_amp_crp': recon_amp_crp,
            })

        target_amp_crp = utils.crop_image(target_amp, target_res, stacked_complex=False)
        return target_amp_crp, all_results


def _namespace_value(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_eval_config(cli_args: argparse.Namespace) -> argparse.Namespace:
    ckpt = torch.load(cli_args.ckpt_path, map_location='cpu')
    hp = ckpt.get('hyper_parameters', {})
    train_args = hp.get('args', hp)

    merged = {
        'model': cli_args.model or _namespace_value(train_args, 'model', 'holonet'),
        'input_mode': cli_args.input_mode or _namespace_value(train_args, 'input_mode', 'real_imag'),
        'channel': cli_args.channel,
        'valid_root': cli_args.valid_root,
        'batch_size': cli_args.batch_size,
        'scale_output': _namespace_value(train_args, 'scale_output', cli_args.scale_output),
        'ckpt_path': cli_args.ckpt_path,
        'save_dir': cli_args.save_dir,
        'ft': cli_args.ft if cli_args.ft is not None else _namespace_value(train_args, 'ft', False),
        'num_workers': cli_args.num_workers,
        'z_start_cm': cli_args.z_start_cm,
        'z_end_cm': cli_args.z_end_cm,
        'z_step_cm': cli_args.z_step_cm,
    }
    return argparse.Namespace(**merged)


def load_phase_generator_weights(system: HoloSystem, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    phase_state = {
        k.replace('phase_generator.', '', 1): v
        for k, v in state_dict.items()
        if k.startswith('phase_generator.')
    }
    missing, unexpected = system.phase_generator.load_state_dict(phase_state, strict=False)
    if missing:
        print(f"[Warn] Missing phase_generator keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected phase_generator keys: {len(unexpected)}")


def save_target_image(save_dir: str, sample_name: str, target_amp_crp: torch.Tensor) -> None:
    target_img = tensor_to_uint8(target_amp_crp[0, 0:1])
    iio.imwrite(os.path.join(save_dir, "target", "all", f"{sample_name}_target.png"), target_img)


def save_poh_and_recon(
    save_dir: str,
    sample_name: str,
    z_cm: int,
    holo_phase: torch.Tensor,
    recon_amp_crp: torch.Tensor,
) -> None:
    """
    Save:
      poh/{z_cm}cm/{sample}_{z_cm}cm_poh.png
      poh/all/{sample}_{z_cm}cm_poh.png
      recon/{z_cm}cm/{sample}_{z_cm}cm_recon.png
      recon/all/{sample}_{z_cm}cm_recon.png
    """
    phase_vis = (holo_phase[0] + torch.pi) / (2 * torch.pi)
    phase_img = tensor_to_uint8(phase_vis)

    recon_img = tensor_to_uint8(recon_amp_crp[0, 0:1])

    poh_name = f"{sample_name}_{z_cm:02d}cm_poh.png"
    recon_name = f"{sample_name}_{z_cm:02d}cm_recon.png"

    # by distance
    iio.imwrite(
        os.path.join(save_dir, "poh", f"{z_cm:02d}cm", poh_name),
        phase_img
    )
    iio.imwrite(
        os.path.join(save_dir, "recon", f"{z_cm:02d}cm", recon_name),
        recon_img
    )


def main():
    cli_args = parse_args()
    args = load_eval_config(cli_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_list_cm = list(range(args.z_start_cm, args.z_end_cm + 1, args.z_step_cm))
    z_list_m = [z * 1e-2 for z in z_list_cm]

    prepare_save_dirs(args.save_dir, z_list_cm)

    system = HoloSystem(args)
    load_phase_generator_weights(system, args.ckpt_path)
    system.eval()
    system.to(device)

    loader = system.test_dataloader()

    print(f"[Info] Loaded checkpoint: {args.ckpt_path}")
    print(f"[Info] Model: {args.model}")
    print(f"[Info] Input mode: {args.input_mode}")
    print(f"[Info] FT: {args.ft}")
    print(f"[Info] Saving results to: {args.save_dir}")
    print(f"[Info] Sweep distances: {z_list_cm[0]}cm ~ {z_list_cm[-1]}cm")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            target_amp, depth = batch
            target_amp = target_amp.to(device, non_blocking=True)

            sample_name = f"sample_{batch_idx:04d}"

            target_amp_crp, results = system.infer_phase_and_recon_sweep(target_amp, z_list_m)

            # save target once
            save_target_image(args.save_dir, sample_name, target_amp_crp)

            # save all distances
            for item, z_cm in zip(results, z_list_cm):
                holo_phase = item['holo_phase']
                recon_amp_crp = item['recon_amp_crp']

                save_poh_and_recon(
                    save_dir=args.save_dir,
                    sample_name=sample_name,
                    z_cm=z_cm,
                    holo_phase=holo_phase,
                    recon_amp_crp=recon_amp_crp,
                )

            print(f"[{batch_idx + 1}/{len(loader)}] saved {sample_name}")

    print("[Done] All results saved successfully.")


if __name__ == '__main__':
    main()
