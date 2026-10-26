import os
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import tools.utils as utils
from tools.loss import SSIMLoss, LPIPSLoss
from tools.propagation.ASM_serial import SerialProp
from tools.propagation.ASM import propagation_ASM
from models.archs.cdfn import CDFN
from models.archs.holonet import HoloNet
from datasets.rgbd import RGBD
import math
import time

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description='CDFN Training Script')
    p.add_argument(
        '--model',
        type=str,
        default='holonet',
        choices=['cdfn', 'holonet'],
        help='Phase generator architecture'
    )
    p.add_argument(
        '--input_mode',
        type=str,
        default='real_imag',
        choices=['real_imag', 'amp_phase'],
        help='Feature representation for real-valued models'
    )
    p.add_argument(
        '--val_dist_samples',
        type=int,
        default=3,
        help='Number of train distances sampled per epoch for multi-distance validation in FT mode'
    )
    p.add_argument(
        '--continuous_train_dist',
        type=utils.str2bool,
        default=True,
        help='Sample training distances continuously inside [min_dist, max_dist] instead of from a fixed list'
    )
    p.add_argument('--channel', type=int, default=1,
                  help='Channel selection: red:0, green:1, blue:2, rgb:3')
    p.add_argument('--train_root', type=str, default='./data/DIV2K_train_HR', help='data root')
    p.add_argument('--valid_root', type=str, default='./data/DIV2K_valid_HR', help='data root')
    p.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    p.add_argument('--batch_size', type=int, default=1, help='Size of minibatch')
    p.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    p.add_argument('--scale_output', type=float, default=0.95,
                  help='Scale of output applied to reconstructed intensity from SLM')
    p.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'composite'],
                  help='mse: manuscript amplitude MSE; composite: original L1/SI-L1/SSIM/LPIPS loss')
    p.add_argument('--loss_l1_weight', type=float, default=0.35, help='Weight for amplitude L1 loss')
    p.add_argument('--loss_si_weight', type=float, default=0.25, help='Weight for scale-invariant amplitude L1 loss')
    p.add_argument('--loss_ssim_weight', type=float, default=0.25, help='Weight for SSIM loss')
    p.add_argument('--loss_lpips_weight', type=float, default=0.15, help='Weight for LPIPS loss')
    p.add_argument('--lpips_net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'], help='LPIPS backbone')
    p.add_argument('--log_dir', type=str, default='tb_logs_2d_loss_paper', help='tensorboard log directory')
    p.add_argument('--exp_name', type=str, default='exp', help='experiment name')
    p.add_argument('--ft', type=utils.str2bool, default=False, help='Use FT')
    p.add_argument('--resume', type=utils.str2bool, default=False, help='Resume')
    p.add_argument('--ckpt_path', type=str, help='checkpoint path')
    p.add_argument(
        '--ft_from_scratch',
        type=utils.str2bool,
        default=False,
        help='When ft=True, do NOT load pretrained, train variable-distance from scratch'
    )
    p.add_argument(
        '--freeze_mode',
        type=str,
        default='none',
        choices=['none', 'adapt_only', 'routing_only', 'all'],
        help='Freeze strategy after loading pretrained model'
    )
    return p.parse_args()

class HoloSystem(LightningModule):
    """
    PyTorch Lightning module for holographic reconstruction system using CDFN.

    Attributes:
        batch_size (int): Size of each training batch
        args (argparse.Namespace): Command line arguments
        channel (int): Selected color channel
        prop_dist (float): Propagation distance
        wavelength (float): Light wavelength
        feature_size (tuple): SLM pitch dimensions
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self._init_system_params(args)
        self._init_propagation_kernels()
        self._init_model_components()
        self._init_datasets()
        self._init_loss_functions()

    def _init_system_params(self, args: argparse.Namespace) -> None:
        """Initialize system parameters."""
        self.batch_size = args.batch_size
        self.args = args
        self.channel = args.channel
        self.ft = args.ft

        # Physical parameters
        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        self.prop_dist = [-200 * mm, -200 * mm, -200 * mm][self.channel]
        self.wavelength = (638 * nm, 520 * nm, 450 * nm)[self.channel]
        self.feature_size = (6.4 * um, 6.4 * um)
        self.pitch = 6.4 * um
        self.pad = False
        self.scale_output = args.scale_output
        self.min_dist = -200 * mm
        self.max_dist = -300 * mm
        self.train_dist_list = np.linspace(self.min_dist, self.max_dist, 11).tolist()
        self.fixed_val_dist = -250 * mm

        # Resolution parameters
        self.homography_res = (1072, 1920)
        self.roi_res = (880, 1600)

    def _init_propagation_kernels(self) -> None:
        """Initialize propagation kernels for the current distance only."""
        self._set_propagation(distance=self.prop_dist)

    def _init_model_components(self) -> None:
        """Initialize neural network components.

        Sets up:
            - CDFN phase generator with IPM and DS options
        """
        if self.args.model == 'cdfn':
            self.phase_generator = CDFN(ft=self.ft, input_mode=self.args.input_mode)
        elif self.args.model == 'holonet':
            self.phase_generator = HoloNet(ft=self.ft, input_mode=self.args.input_mode)
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

    def _init_datasets(self) -> None:
        """Initialize training and validation datasets."""
        self.train_set = RGBD(
            self.args.train_root,
            channel=self.channel,
            split='train',
            image_res=self.roi_res,
            homography_res=self.homography_res
        )
        self.val_set = RGBD(
            self.args.valid_root,
            channel=self.channel,
            split='val',
            image_res=self.roi_res,
            homography_res=self.homography_res
        )

    def _init_loss_functions(self) -> None:
        """Use manuscript amplitude MSE, with the original loss available."""
        # Namespaces stored in old checkpoints have no loss_type.
        self.loss_type = getattr(self.args, 'loss_type', 'composite')
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11)
        self.use_lpips_loss = self.loss_type == 'composite' and self.args.loss_lpips_weight > 0
        if self.use_lpips_loss:
            self.lpips_loss = LPIPSLoss(net=self.args.lpips_net)
        else:
            self.lpips_loss = None

    def _get_random_distance(self):
        """Get random propagation distance from pre-computed distances."""
        if self.args.continuous_train_dist:
            lo = min(self.min_dist, self.max_dist)
            hi = max(self.min_dist, self.max_dist)
            return float(np.random.uniform(lo, hi))
        return float(np.random.choice(self.train_dist_list))

    def _get_val_sampled_distances(self):
        if not self.ft:
            return [self.prop_dist]

        if self.args.val_dist_samples <= 0:
            return [self.fixed_val_dist]

        candidates = [float(d) for d in self.train_dist_list if d != self.fixed_val_dist]
        if not candidates:
            return [self.fixed_val_dist]

        sample_count = min(self.args.val_dist_samples, len(candidates))
        start_idx = self.current_epoch % len(candidates)
        sampled = [candidates[(start_idx + i) % len(candidates)] for i in range(sample_count)]
        return [self.fixed_val_dist] + sampled

    def _build_propagation_bundle(self, distance: float) -> Dict[str, torch.Tensor]:
        n, m = self.homography_res
        field = torch.empty(self.batch_size, 1, n, m)
        return {
            'Hforward_H': propagation_ASM(
                field,
                feature_size=[self.pitch, self.pitch],
                wavelength=self.wavelength,
                z=distance,
                linear_conv=self.pad,
                return_H=True
            ),
            'routing_H': propagation_ASM(
                field,
                feature_size=[self.pitch, self.pitch],
                wavelength=self.wavelength,
                z=distance,
                linear_conv=self.pad,
                return_H=True,
                band_lm=False
            ),
            'Hbackward_H': propagation_ASM(
                field,
                feature_size=[self.pitch, self.pitch],
                wavelength=self.wavelength,
                z=-distance,
                linear_conv=self.pad,
                return_H=True
            ),
        }

    def _set_propagation(self, distance: float) -> None:
        self.prop_dist = float(distance)
        bundle = self._build_propagation_bundle(self.prop_dist)
        self.Hforward_H = bundle['Hforward_H'].to(self.device)
        self.H = bundle['routing_H'].to(self.device)
        self.Hbackward_H = bundle['Hbackward_H'].to(self.device)

    def _compute_recon_loss(self, recon_amp_crp: torch.Tensor, target_amp_crp: torch.Tensor):
        if self.loss_type == 'mse':
            # Equation (loss): MSE(A, A'). No fitted brightness scale,
            # clipping, or perceptual term is applied to the objective.
            loss_mse = self.mse_loss(recon_amp_crp, target_amp_crp)
            zero = recon_amp_crp.new_zeros(())
            metrics = {
                'mse': loss_mse,
                'l1': zero, 'si_l1': zero, 'ssim': zero, 'lpips': zero,
                'pred_mean': recon_amp_crp.square().mean(),
                'target_mean': target_amp_crp.square().mean(),
            }
            return loss_mse, metrics, recon_amp_crp

        reduce_dims = tuple(range(1, recon_amp_crp.ndim))

        alpha = (recon_amp_crp * target_amp_crp).sum(dim=reduce_dims, keepdim=True) / (
            recon_amp_crp.square().sum(dim=reduce_dims, keepdim=True) + 1e-8
        )
        recon_amp_scaled = alpha * recon_amp_crp

        recon_int_scaled = recon_amp_scaled.square()
        target_int = target_amp_crp.square()

        loss_l1 = self.l1_loss(recon_amp_crp, target_amp_crp)
        loss_si = self.l1_loss(recon_amp_scaled, target_amp_crp)
        loss_ssim = self.ssim_loss(recon_amp_scaled, target_amp_crp)

        if self.lpips_loss is not None:
            loss_lpips = self.lpips_loss(recon_amp_scaled, target_amp_crp)
        else:
            loss_lpips = recon_amp_crp.new_zeros(())

        loss_total = (
            self.args.loss_l1_weight * loss_l1
            + self.args.loss_si_weight * loss_si
            + self.args.loss_ssim_weight * loss_ssim
            + self.args.loss_lpips_weight * loss_lpips
        )

        metrics = {
            'mse': self.mse_loss(recon_amp_crp, target_amp_crp),
            'l1': loss_l1,
            'si_l1': loss_si,
            'ssim': loss_ssim,
            'lpips': loss_lpips,
            'pred_mean': recon_int_scaled.mean(),
            'target_mean': target_int.mean(),
        }
        return loss_total, metrics, recon_amp_scaled

    def _forward_at_distance(self, target_amp: torch.Tensor, distance: float):
        self._set_propagation(distance)
        target_res = self.roi_res
        U_new = target_amp * torch.exp(1j * torch.zeros_like(target_amp))
        slmfield, holo_phase = self.phase_generator(
            U_new, self.Hbackward_H, distance=(self.H, self.prop_dist)
        )
        holo_cpx = torch.exp(1j * holo_phase)
        recon_complex = propagation_ASM(holo_cpx, precomped_H=self.Hforward_H)
        recon_amp = recon_complex.abs() * math.sqrt(self.scale_output)
        recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)
        target_amp_crp = utils.crop_image(target_amp, target_res, stacked_complex=False)
        loss, loss_metrics, recon_amp_scaled = self._compute_recon_loss(recon_amp_crp, target_amp_crp)
        return loss, recon_amp_crp, recon_amp_scaled, holo_phase, loss_metrics


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Execute one training step."""
        target_amp, depth = batch

        if self.ft:
            self._set_propagation(self._get_random_distance())
        else:
            self._set_propagation(self.prop_dist)

        loss_main, _, _, _, loss_metrics = self._forward_at_distance(target_amp, self.prop_dist)

        # Log metrics
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else self.args.lr
        log = {
            'train/loss': loss_main,
            'train/loss_mse': loss_metrics['mse'],
            'train/loss_l1': loss_metrics['l1'],
            'train/loss_si_l1': loss_metrics['si_l1'],
            'train/loss_ssim': loss_metrics['ssim'],
            'train/loss_lpips': loss_metrics['lpips'],
            'train/pred_mean': loss_metrics['pred_mean'],
            'train/target_mean': loss_metrics['target_mean'],
            'train/prop_dist_mm': abs(self.prop_dist) * 1e3,
            'train/lr': current_lr,
        }
        self.log_dict(log, prog_bar=False, on_step=True, on_epoch=True, batch_size=target_amp.size(0))
        return {'loss': loss_main, 'log': log}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Execute one validation step."""
        target_amp, depth = batch
        eval_distances = self._get_val_sampled_distances()
        loss_main, recon_amp_crp, recon_amp_scaled, holo_phase, loss_metrics = self._forward_at_distance(target_amp, eval_distances[0])
        target_amp_crp = utils.crop_image(target_amp, self.roi_res, stacked_complex=False)

        multi_losses = [loss_main]
        for distance in eval_distances[1:]:
            dist_loss, _, _, _, _ = self._forward_at_distance(target_amp, distance)
            multi_losses.append(dist_loss)

        loss_multi = torch.stack(multi_losses).mean()

        # Log metrics
        log = {
            'val/loss': loss_main,
            'val_loss': loss_main,
            'val/loss_multi': loss_multi,
            'val_loss_multi': loss_multi,
            'val/loss_mse': loss_metrics['mse'],
            'val/loss_l1': loss_metrics['l1'],
            'val/loss_si_l1': loss_metrics['si_l1'],
            'val/loss_ssim': loss_metrics['ssim'],
            'val/loss_lpips': loss_metrics['lpips'],
            'val/pred_mean': loss_metrics['pred_mean'],
            'val/target_mean': loss_metrics['target_mean'],
            'val/prop_dist_mm': abs(eval_distances[0]) * 1e3,
        }
        self.log_dict(log, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True, batch_size=target_amp.size(0))

        # Log images periodically
        if batch_idx == 0:
            self.logger.experiment.add_image(
                'phases',
                (holo_phase[0, ...] + torch.pi) / (2 * torch.pi),
                self.current_epoch
            )
            self.logger.experiment.add_image(
                    f'recon_amp_crp',
                    recon_amp_crp[0, 0:1, ...] / (recon_amp_crp[0, 0:1, ...].amax() + 1e-8),
                    self.current_epoch
                )
            self.logger.experiment.add_image(
                    f'recon_amp_crp_scaled',
                    recon_amp_scaled[0, 0:1, ...] / (recon_amp_scaled[0, 0:1, ...].amax() + 1e-8),
                    self.current_epoch
                )
            self.logger.experiment.add_image(
                    f'target_amp_crp',
                    target_amp_crp[0, 0:1, ...] / (target_amp_crp[0, 0:1, ...].amax() + 1e-8),
                    self.current_epoch
                )

        return {'loss': loss_multi, 'log': log}

    def train_dataloader(self) -> DataLoader:
        """Configure training data loader.

        Returns:
            DataLoader: Training data loader with specified batch size and settings
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Configure validation data loader.

        Returns:
            DataLoader: Validation data loader with specified batch size and settings
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Configure test data loader.

        Returns:
            DataLoader: Test data loader with batch size 1 for evaluation
        """
        return DataLoader(
            self.val_set,
            batch_size=1,  # Use batch size 1 for testing
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure model optimizer.

        Returns:
            Optimizer: Adam optimizer with specified learning rate
        """
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.num_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def freeze_pretrained_params(system, freeze_mode='adapt_only'):
    """
    freeze_mode:
        - 'none': 不冻结
        - 'adapt_only': 只训练 adapt 分支，其余 phase_generator 参数冻结
        - 'routing_only': 只训练 adapt.routing 分支
        - 'all': 冻结整个 phase_generator（通常不适合训练）
    """
    frozen_numel = 0
    trainable_numel = 0
    frozen_names = []
    trainable_names = []

    for name, param in system.named_parameters():
        # 默认全部可训练
        param.requires_grad = True

        if name.startswith("phase_generator."):
            if freeze_mode == 'none':
                param.requires_grad = True

            elif freeze_mode == 'all':
                param.requires_grad = False

            elif freeze_mode == 'adapt_only':
                if ".adapt." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            elif freeze_mode == 'routing_only':
                if ".adapt.routing." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            else:
                raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

        if param.requires_grad:
            trainable_numel += param.numel()
            trainable_names.append(name)
        else:
            frozen_numel += param.numel()
            frozen_names.append(name)

    print(f"[Info] Freeze mode: {freeze_mode}")
    print(f"[Info] Frozen params: {frozen_numel}")
    print(f"[Info] Trainable params: {trainable_numel}")

    print("[Info] First 20 trainable parameter names:")
    for n in trainable_names[:20]:
        print("  ", n)

    print("[Info] First 20 frozen parameter names:")
    for n in frozen_names[:20]:
        print("  ", n)

def main():
    """Main training function."""
    args = parse_args()

    # -----------------------------
    # basic sanity checks
    # -----------------------------
    if (not args.ft) and args.ft_from_scratch:
        raise ValueError("--ft_from_scratch only works when --ft true")

    if args.resume and not args.ckpt_path:
        raise ValueError("When --resume true, you must provide --ckpt_path")

    if args.freeze_mode != 'none':
        if not args.ft:
            raise ValueError("--freeze_mode only works when --ft true")
        if args.ft_from_scratch:
            raise ValueError("--freeze_mode requires loading pretrained weights, so it cannot be used with --ft_from_scratch true")

    # 可选：提升 3090 Tensor Core 利用率
    torch.set_float32_matmul_precision('high')

    # -----------------------------
    # logger / checkpoint
    # -----------------------------
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name
    )

    print(f"[Info] Model: {args.model}")
    print(f"[Info] Input mode: {args.input_mode}")
    monitor_metric = 'val_loss_multi' if args.ft else 'val_loss'
    filename_pattern = '{epoch:02d}-{val_loss_multi:.6f}' if args.ft else '{epoch:02d}-{val_loss:.6f}'
    print(f"[Info] Checkpoint monitor: {monitor_metric}")

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename=filename_pattern,
        save_top_k=1,
        save_last=True,
        mode='min',
        auto_insert_metric_name=False
    )

    # -----------------------------
    # build system
    # -----------------------------
    if args.resume:
        # 断点续训：恢复整个 lightning module + optimizer/scheduler state
        print(f"[Info] Resume training from checkpoint: {args.ckpt_path}")

        system = HoloSystem.load_from_checkpoint(
            args.ckpt_path,
            args=args
        )

    else:
        # 新实验
        system = HoloSystem(args)

        # ---------- fixed-distance training ----------
        if not args.ft:
            print("[Info] Fixed-distance training from scratch")

        # ---------- variable-distance training ----------
        else:
            # 变距离从头训练
            if args.ft_from_scratch:
                print("[Info] Variable-distance training from scratch (no pretrained loaded)")

            # 变距离 + 载入固定距离预训练
            else:
                if not args.ckpt_path:
                    raise ValueError(
                        "When --ft true and --ft_from_scratch false, you must provide --ckpt_path"
                    )

                print(f"[Info] FT mode, load pretrained from: {args.ckpt_path}")

                ckpt = torch.load(args.ckpt_path, map_location='cpu')
                if 'state_dict' not in ckpt:
                    raise KeyError(f"No 'state_dict' found in checkpoint: {args.ckpt_path}")

                pretrained_state = ckpt['state_dict']
                current_state = system.state_dict()

                load_dict = {}
                loaded_keys = []
                skipped_keys = []

                for k, v in pretrained_state.items():
                    # 只加载 phase_generator 的权重
                    if not k.startswith("phase_generator."):
                        continue

                    if k in current_state and current_state[k].shape == v.shape:
                        load_dict[k] = v
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(k)

                current_state.update(load_dict)
                system.load_state_dict(current_state, strict=False)

                print(f"[Info] Loaded params: {len(loaded_keys)}")
                print(f"[Info] Skipped params: {len(skipped_keys)}")
                if len(skipped_keys) > 0:
                    print("[Info] Example skipped keys:")
                    for k in skipped_keys[:10]:
                        print("   ", k)

                # ---------- freeze strategy ----------
                if args.freeze_mode != 'none':
                    print(f"[Info] Applying freeze mode: {args.freeze_mode}")
                    freeze_pretrained_params(system, freeze_mode=args.freeze_mode)

    # -----------------------------
    # print trainable summary
    # -----------------------------
    total_params = 0
    trainable_params = 0
    for p in system.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n

    print(f"[Info] Total params: {total_params}")
    print(f"[Info] Trainable params after setup: {trainable_params}")

    if trainable_params == 0:
        raise ValueError(
            "No trainable parameters remain after setup. "
            "Please check --freeze_mode."
        )

    # -----------------------------
    # trainer
    # -----------------------------
    trainer = Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        strategy='auto',
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=4,
    )

    # -----------------------------
    # fit
    # -----------------------------
    if args.resume:
        trainer.fit(system, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(system)


if __name__ == '__main__':
    main()
