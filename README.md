# High-Quality Depth-Adaptive Computer-Generated Hologram Using Depth-aware Plug-in Network

Official implementation of **High-Quality Depth-Adaptive Computer-Generated Hologram Using Depth-aware Plug-in Network**.

**Ganzhangqin Yuan and Zihan Geng**

## Code

The repository is currently being organized. The source code will be released here soon.

The documentation below describes the current implementation and its training and inference workflows for the upcoming code release.

## Overview

This project implements a depth-aware plug-in network for phase-only computer-generated holography. The plug-in adapts intermediate network features to the reconstruction distance through dynamic convolution. A routing network encodes the phase of the optical transfer function (OTF) and predicts weights that combine four learnable convolution kernels. The resulting feature modulation is applied through a residual connection.

The current implementation includes:

- HoloNet and CDFN phase generators, with optional depth-aware plug-in modules.
- Fixed-distance pretraining and variable-distance adaptation, including training only the plug-in parameters.
- Differentiable angular spectrum method (ASM) propagation.
- Amplitude mean squared error (MSE) training, with an optional composite reconstruction loss.
- Reconstruction distance sweeps with phase and reconstructed-amplitude image export.

Here, depth adaptation refers to the reconstruction-plane distance. Although the dataset loader reads paired RGB and depth images, the current training and inference scripts use the target amplitude and propagation transfer functions; they do not use the loaded depth maps as network inputs.

## Repository Structure

```text
.
├── main_2d_diffdn.py             # Training and validation
├── test_2d_diffdn.py             # Distance-sweep inference and image export
├── requirements.txt             # Pinned Python dependencies
├── datasets/
│   └── rgbd.py                   # Paired RGB/depth loading and preprocessing
├── models/
│   ├── archs/                   # HoloNet and CDFN phase generators
│   ├── backbones/               # U-Net and residual U-Net backbones
│   └── components/
│       ├── ftmodule.py          # Dynamic convolution and residual adaptation
│       └── transformerz.py      # OTF-phase routing network
└── tools/
    ├── loss.py                  # Reconstruction losses
    ├── utils.py                 # Image and tensor utilities
    └── propagation/             # ASM propagation implementations
```

## Installation

The dependency file records Python **3.8.18**, PyTorch **2.1.0**, torchvision **0.16.0**, and PyTorch Lightning **2.1.1** as the reference environment. A CUDA-capable GPU is recommended for the default image resolution.

Create an environment and run the following commands from the repository root:

```bash
conda create -n depth-aware-cgh python=3.8.18 -y
conda activate depth-aware-cgh
python -m pip install -r requirements.txt
```

For GPU execution, install PyTorch 2.1.0 and torchvision 0.16.0 builds compatible with your CUDA environment before installing the remaining requirements. The default MSE objective does not require pretrained perceptual weights; enabling the LPIPS loss may download its backbone weights on first use.

## Data Preparation

The default paths expect the following layout:

```text
data/
├── DIV2K_train_HR/
│   ├── rgb/
│   │   ├── 0001.png
│   │   └── ...
│   └── depth8bit/
│       ├── 0001.png
│       └── ...
└── DIV2K_valid_HR/
    ├── rgb/
    │   ├── 0801.png
    │   └── ...
    └── depth8bit/
        ├── 0801.png
        └── ...
```

Prepare RGB images and corresponding 8-bit depth images separately. Both directories are required by the loader, even though the current workflows do not use depth values. Files are paired by their independently sorted filenames, so the directories must contain the same number of images in matching order. Keep only image files in these directories.

Custom dataset locations can be supplied through `--train_root` and `--valid_root`. RGB intensities are linearized and converted to amplitude. Images are cropped to the target aspect ratio, resized to an **880 × 1600** region of interest, and padded to **1072 × 1920** for propagation. Resolutions are listed as height × width.

## Training

### 1. Fixed-distance pretraining

Train a baseline phase generator at a reconstruction distance of 20 cm:

```bash
python main_2d_diffdn.py \
    --model holonet \
    --input_mode real_imag \
    --channel 1 \
    --train_root ./data/DIV2K_train_HR \
    --valid_root ./data/DIV2K_valid_HR \
    --ft false \
    --loss_type mse \
    --batch_size 1 \
    --num_epochs 20 \
    --lr 5e-5 \
    --exp_name holonet_fixed
```

Replace `--model holonet` with `--model cdfn` to use the CDFN implementation.

### 2. Depth-adaptive plug-in training

Initialize from a fixed-distance checkpoint and train only the plug-in parameters:

```bash
python main_2d_diffdn.py \
    --model holonet \
    --input_mode real_imag \
    --channel 1 \
    --train_root ./data/DIV2K_train_HR \
    --valid_root ./data/DIV2K_valid_HR \
    --ft true \
    --ckpt_path /path/to/fixed_distance.ckpt \
    --freeze_mode adapt_only \
    --continuous_train_dist true \
    --loss_type mse \
    --batch_size 1 \
    --num_epochs 20 \
    --lr 5e-5 \
    --exp_name holonet_depth_adaptive
```

Use the same architecture, feature representation, and color channel as the pretrained model. In this mode, training samples reconstruction distances continuously between 20 and 30 cm. Validation evaluates 25 cm plus three additional distances selected from the discrete training-distance grid by default.

Available parameter-freezing strategies are:

| `--freeze_mode` | Trainable phase-generator parameters |
| --- | --- |
| `none` | All parameters; this is the default |
| `adapt_only` | Plug-in kernel pools and routing networks |
| `routing_only` | Plug-in routing networks only |

To train the depth-adaptive model from scratch, use `--ft true --ft_from_scratch true`, omit `--ckpt_path`, and keep `--freeze_mode none`. To continue an interrupted run, use `--resume true --ckpt_path /path/to/last.ckpt` together with the original model and training options. `--num_epochs` specifies the total target epoch count when resuming.

### Training options and outputs

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `holonet` | Phase generator: `holonet` or `cdfn` |
| `--input_mode` | `real_imag` | Complex-field features: `real_imag` or `amp_phase` |
| `--channel` | `1` | Single color channel: `0` red, `1` green, `2` blue |
| `--ft` | `false` | Enable plug-in modules and variable-distance training |
| `--batch_size` | `1` | Must be `1` when plug-in modules are enabled |
| `--continuous_train_dist` | `true` | Sample continuously; `false` selects from 11 distances at 1 cm intervals |
| `--loss_type` | `mse` | Amplitude MSE; `composite` combines L1, scale-invariant L1, SSIM, and LPIPS |
| `--val_dist_samples` | `3` | Additional validation distances in plug-in mode; `0` uses only 25 cm |
| `--scale_output` | `0.95` | Intensity scaling factor; reconstructed amplitude is multiplied by its square root |
| `--log_dir` | `tb_logs_2d_loss_paper` | TensorBoard and checkpoint root |
| `--exp_name` | `exp` | Experiment name |

The trainer uses AdamW, cosine learning-rate scheduling, gradient clipping, and gradient accumulation over four batches. Each run saves the best validation checkpoint and `last.ckpt` under:

```text
tb_logs_2d_loss_paper/<exp_name>/version_<n>/checkpoints/
```

Inspect training curves and validation images with:

```bash
tensorboard --logdir tb_logs_2d_loss_paper
```

## Inference

Run a reconstruction sweep from 20 to 30 cm in 1 cm steps:

```bash
python test_2d_diffdn.py \
    --channel 1 \
    --valid_root ./data/DIV2K_valid_HR \
    --ckpt_path /path/to/depth_adaptive.ckpt \
    --save_dir ./test_results_sweep \
    --batch_size 1 \
    --z_start_cm 20 \
    --z_end_cm 30 \
    --z_step_cm 1
```

The script restores `model`, `input_mode`, and `ft` from the checkpoint configuration unless explicitly overridden. Set `--channel` to the channel used during training. It uses CUDA when available and otherwise runs on CPU. A separate hologram is generated for each requested distance.

Results are saved as:

```text
test_results_sweep/
├── poh/
│   ├── 20cm/                    # Phase-only hologram visualizations
│   ├── ...
│   └── 30cm/
├── recon/
│   ├── 20cm/                    # Reconstructed amplitude visualizations
│   ├── ...
│   └── 30cm/
└── target/
    └── all/                     # Target amplitude visualizations
```

The exporter also creates `poh/all/` and `recon/all/`, but currently leaves them empty. Exported PNG images are individually min–max normalized for visualization; they do not preserve absolute amplitude or a calibrated phase-to-gray mapping. The sweep script does not compute quantitative reconstruction metrics.

## Current Implementation Notes

- **Single-channel operation:** use `--channel 0`, `1`, or `2`. Although the CLI help mentions `3` for RGB, the current system parameters support only individual channels. Process channels separately for color experiments.
- **Plug-in batch size:** dynamic convolution currently supports only batch size `1` for both training and inference.
- **Optical settings:** wavelengths are 638 nm, 520 nm, and 450 nm for red, green, and blue, respectively; the SLM pixel pitch is 6.4 μm. Optical parameters, resolutions, and the training distance range are configured in `_init_system_params` in the training and inference scripts.

## Citation

If you use this implementation in your research, please cite:

> Ganzhangqin Yuan and Zihan Geng. *High-Quality Depth-Adaptive Computer-Generated Hologram Using Depth-aware Plug-in Network*.

Publication details and a BibTeX entry will be added when available.

## Acknowledgments and Licensing

The ASM propagation and utility code include attribution to:

> Yifan Peng, Suyeon Choi, Nitish Padmanaban, and Gordon Wetzstein. *Neural Holography with Camera-in-the-loop Training*. ACM Transactions on Graphics (SIGGRAPH Asia), 2020.

The headers in [`tools/propagation/ASM.py`](tools/propagation/ASM.py) and [`tools/utils.py`](tools/utils.py) specify Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) terms for that code. Retain these notices and the associated attribution when redistributing those files.
