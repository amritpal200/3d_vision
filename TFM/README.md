# TFM Model — SSIM Loss Improvement

## Overview

This is a modified version of the original `TFM_model.py` from [M3D-VTON](https://github.com/fyviezhao/M3D-VTON). The modification adds **Structural Similarity Index Measure (SSIM) loss** as a complementary training signal to improve texture quality in the generated try-on images.

---

## What was changed

The original TFM used the following loss function:

```
loss_G = L1 + VGG + mask
```

We extended it to:

```
loss_G = L1 + VGG + mask + SSIM
```

where `SSIM = 0.5 * (1 - ssim(p_tryon, im))`.

### Specific changes in the code

**1. New import at the top of the file:**
```python
from pytorch_msssim import SSIM
```

**2. In `__init__`, updated loss names and added SSIM criterion:**
```python
self.loss_names = ['G', 'l1', 'vgg', 'ssim', 'mask']
self.criterionSSIM = SSIM(data_range=1.0, size_average=True, channel=3)
```

**3. In `backward_G()`, added SSIM loss computation and included it in `loss_G`:**
```python
self.loss_ssim = 0.5 * (1 - self.criterionSSIM(
    (self.p_tryon + 1) / 2,
    (self.im + 1) / 2
))
self.loss_G = self.loss_l1 + self.loss_vgg + self.loss_mask + self.loss_ssim
```

---

## Why SSIM

The original L1 loss compares images pixel by pixel, which tends to produce blurry outputs on garments with complex textures such as checkered patterns, stripes, or floral prints. SSIM measures structural similarity by comparing local windows of the image — luminance, contrast and structure — explicitly penalizing the loss of local structural coherence. This forces the model to preserve sharper edges and more faithful texture details.

---

## Installation

Install the required dependency:

```bash
pip install pytorch-msssim
```

---

## Results

The model was trained on the MPV3D dataset (5632 training pairs, 100 epochs) using an NVIDIA L40S GPU with 48GB VRAM, Adam optimizer, learning rate 0.0001, batch size 8, and loss weights λ_L1 = λ_VGG = λ_mask = 1.0, λ_SSIM = 0.5.

| Model | FID ↓ | SSIM ↑ |
|-------|-------|--------|
| Original paper | 20.04 | 0.8804 |
| Baseline (our reproduction) | 16.487 | 0.9232 |
| **TFM + SSIM (ours, 100 epochs)** | **15.950** | **0.9249** |
| TFM + SSIM (ours, 200 epochs) | 16.201 | - |

---

## Training command

```bash
python train.py --model TFM --name TFM_ssim \
    --dataroot data/MPV3D/MPV3D \
    --warproot results/aligned/MTM/train_pairs \
    --datalist train_pairs \
    --checkpoints_dir checkpoints
```

## Test command

```bash
python test.py --model TFM --name TFM_ssim \
    --dataroot data/MPV3D/MPV3D \
    --warproot results_test/aligned/MTM/test_pairs \
    --datalist test_pairs \
    --checkpoints_dir checkpoints \
    --results_dir results_ssim
```

---

## Notes

- No architectural changes were made — only the loss function was modified.
- The SSIM loss adds no computational overhead at inference time since it is only used during training.
- The backup of the original file is available as `TFM_model_backup.py`.