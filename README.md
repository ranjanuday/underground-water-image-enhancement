# Underwater Image Enhancement Using Deep Learning (UIE-Net)

## Overview

UIE-Net is a deep learning-based framework designed to enhance underwater images by addressing three major challenges:

- Color distortion
- Haze caused by light scattering
- Loss of fine details and textures

The proposed architecture combines three specialized branches:

1. Color Correction Branch
2. Dehazing Branch
3. U-Net Detail Enhancement Branch

The outputs from these branches are fused through a Spatial Fusion Module to generate a visually enhanced underwater image.

---

## Problem Statement

Underwater images often suffer from:

- Blue-green color dominance due to wavelength-dependent absorption
- Reduced visibility because of scattering and haze
- Loss of structural details and textures
- Non-uniform illumination

Most existing methods focus on only one or two of these issues. UIE-Net aims to solve all of them simultaneously.

---

## Proposed Architecture

UIE-Net consists of:

### 1. Color Correction Branch
Restores natural underwater colors and reduces color casts.

### 2. Dehazing Branch
Uses atmospheric scattering principles to remove underwater haze and improve visibility.

### 3. U-Net Enhancement Branch
An encoder-decoder network that restores textures and sharp details.

### 4. Spatial Fusion Module
Learns adaptive pixel-wise weights to combine outputs from all branches and generate the final enhanced image.

---

## Architecture Diagram

> Add your architecture diagram below.

<p align="center">
  <img src="https://github.com/ranjanuday/underground-water-image-enhancement/blob/main/Architecture%20diagram.png?raw=true" alt="UIE-Net Architecture" width="900"/>
</p>

---

## Dataset

### UIEB Dataset

The model is trained on the UIEB (Underwater Image Enhancement Benchmark) dataset.

- Total Images: 890
- Training Images: 800
- Testing Images: 90

Dataset Characteristics:

- Various underwater depths
- Different turbidity levels
- Diverse illumination conditions

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 100 |
| Batch Size | 16 |
| Image Size | 256 × 256 |
| Learning Rate | 1e-4 |

---

## Results

### Qualitative Results

The model successfully:

- Restores natural colors
- Removes haze
- Improves visibility
- Enhances fine details and textures

### Sample Outputs

| Input | Enhanced Output |
|---------|---------|
| Add Image | Add Image |

---

## Quantitative Results

| Method | PSNR ↑ | SSIM ↑ |
|----------|----------|----------|
| UIEC²-Net | 20.54 | 0.849 |
| Water-Net | 20.18 | 0.812 |
| FUnIE-GAN | 21.02 | 0.824 |
| **UIE-Net (Ours)** | **21.27** | **0.888** |

Performance Improvements:

- +0.73 dB PSNR over UIEC²-Net
- Highest SSIM among compared methods

---

## Project Structure

```text
underwater-image-enhancement/
│
├── dataset/
├── models/
├── training/
├── inference/
├── results/
├── images/
│   └── architecture.png
├── requirements.txt
├── train.py
├── test.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ranjanuday/underground-water-image-enhancement.git
cd underground-water-image-enhancement
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Testing

```bash
python test.py
```

---

## Future Work

- Real-time underwater video enhancement
- Temporal consistency for video frames
- Improved transmission map estimation
- Better handling of dense haze and varying water turbidity

---

---

## References

- Li et al., UIEB Dataset (2020)
- Wang et al., UIEC²-Net (2021)
- Islam et al., FUnIE-GAN (2020)
- Cai et al., DehazeNet (2016)
