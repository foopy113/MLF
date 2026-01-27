# MLFMamba

## A Multi-Level Feature Modelling Framework for Hyperspectral Image Classification

This repository provides the official PyTorch implementation of the paper:


---

## Overview

Hyperspectral images exhibit complex spatial–spectral characteristics, involving both fine-grained spatial structures and highly correlated spectral information.  
To address these challenges efficiently, we propose **MLFMamba**, a multi-level feature modelling framework built upon **Mamba-based state-space models (SSMs)**.

The framework consists of:

- Shallow feature refinement via a **Feature Aware (FA)** module;
- Spatial structure modelling using **Spatial Multi-Scan Mamba (SpaMSM)**;
- Spectral dependency modelling through **Spectral Grouping Bidirectional Mamba (SpeGBM)**;
- Adaptive spatial–spectral fusion for balanced performance and efficiency.

---

## Environment

This code has been tested under the following environment:

- **OS**: Ubuntu 22.04  
- **Python**: 3.10  
- **PyTorch**: 2.1.2  
- **CUDA**: 11.8  
- **GPU**: NVIDIA GPU (≥ 12GB VRAM recommended)

⚠️ **Note**  
This implementation relies on CUDA kernels provided by `mamba-ssm` (i.e., `selective_scan_fn`).  

---

## Installation

### Create a virtual environment (recommended)

```bash
conda create -n mlfmamba python=3.10 -y
conda activate mlfmamba
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Datasets

The datasets used in this paper are publicly available and can be downloaded from the following links:

- **Baidu Netdisk**  
  https://pan.baidu.com/s/1ycCMyI6RrzRjZdrNH51P6g  
  Access code: `acnv`

- **Google Drive**  
  https://drive.google.com/file/d/10TUtGD8q_qpM4l5kQPM3Dmye0WzBRU-N/view?usp=drive_link

After downloading, please organize the data into the following directory structure:

```text
data/
├── IP/        # Indian Pines
├── UP/        # Pavia University
├── LongKou/   # WHU-Hi-LongKou
└── Houston/   # Houston2013
```
Each dataset directory should contain the hyperspectral image data and the corresponding ground-truth labels in `.mat` format.

---

## Running the Code

Training and testing can be conducted using the unified script `train_MLFMamba.py`.

```bash
python train_MLFMamba.py --dataset_index 0   # Indian Pines
python train_MLFMamba.py --dataset_index 1   # Pavia University
python train_MLFMamba.py --dataset_index 2   # WHU-Hi-LongKou
python train_MLFMamba.py --dataset_index 3   # Houston2013
```
The argument `--dataset_index` specifies the dataset to be used for training and evaluation.

---

## Key Modules

### Feature Aware (FA) Module

The Feature Aware (FA) module is designed to refine shallow features before deep modelling.  
It includes:

- Channel mapping via **1×1 pointwise convolution** to adjust feature dimensions;
- Parallel **multi-scale depthwise convolutions** (3×3 / 5×5 / 7×7) to capture spatial patterns under different receptive fields;
- **Residual connections** to preserve original information and stabilize optimization.

---

### Spatial Multi-Scan Mamba (SpaMSM)

The Spatial Multi-Scan Mamba (SpaMSM) module addresses the limitations of conventional single-path scanning strategies in vision Mamba models.

It incorporates:

- **Parallel serpentine scanning** along rows and columns;
- **Diagonal serpentine scanning** to model cross-directional dependencies.

Multiple spatial sequences are modelled and fused, enabling comprehensive two-dimensional spatial context modelling.

---

### Spectral Grouping Bidirectional Mamba (SpeGBM)

The Spectral Grouping Bidirectional Mamba (SpeGBM) module is designed to efficiently exploit the highly correlated and redundant spectral information in hyperspectral images.

Key characteristics include:

- Grouping contiguous spectral bands into multiple spectral subgroups to reduce modelling complexity;
- Performing **forward and backward state-space modelling** for each spectral group;
- Fusing bidirectional representations to capture complete spectral dependencies.





