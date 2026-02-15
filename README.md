# IITB_GNR638 - Assignment 1 (Group 61 [G16])

## Overview
This repository contains our custom deep learning framework for image classification with:
- Tensor + autograd-style parameter handling
- Layers from scratch (`Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `Linear`)
- `CrossEntropyLoss` and SGD optimizer
- CUDA backend (`ctypes` + `.cu` kernels) with Python fallback
- Training, evaluation, serialization, and metrics logging

The implementation is in `assignment1/`.

## What We Are Doing
We train CNN models on two datasets:
- `data_1` (10 classes)
- `data_2` (100 classes)

We use a deterministic split with:
- `seed = 42`
- `train/val/test = 80/10/10`

## Folder Structure
- `assignment1/train.py` - training script
- `assignment1/evaluate.py` - evaluation script
- `assignment1/gnr_dl/` - framework code
- `assignment1/cuda_backend/` - CUDA backend
- `assignment1/config/` - config files for Dataset A/B
- `assignment1/artifacts/` - saved weights and metrics

## Dataset Preparation
Place datasets inside `assignment1/datasets/` as:

```text
assignment1/datasets/
  data_1/
    <class_name>/*.png
  data_2/
    <class_name>/*.png
```

Notes:
- For `data_1`, class folders are typically `0..9`.
- For `data_2`, class folders are CIFAR-100 class names.
- PNG files are expected.

## Environment
- Python 3.10+ (tested with Python 3.12)
- OpenCV (`cv2`)
- CUDA toolkit (`nvcc`) for GPU backend

## Reproducing Results
Run all commands from the repository root (`IITB_GNR638`).

### 1) Build CUDA backend
```bash
cd assignment1/cuda_backend
./build.sh
cd ../..
```

### 2) Optional backend check
```bash
cd assignment1
python3 -c "from gnr_dl.backend import get_backend; print('Backend:', 'CUDA' if get_backend().G16UsingCuda else 'Python fallback')"
cd ..
```

### 3) Dataset A (`data_1`) run
```bash
cd assignment1
python3 train.py \
  --train_dir datasets/data_1 \
  --config config/data1_arch_config.json \
  --weights_out artifacts/weights_data1_split.json \
  --metrics_out artifacts/weights_data1_split_metrics.json

python3 evaluate.py \
  --test_dir datasets/data_1 \
  --config config/data1_arch_config.json \
  --weights artifacts/weights_data1_split_best.json
cd ..
```

### 4) Dataset B (`data_2`) run
```bash
cd assignment1
python3 train.py \
  --train_dir datasets/data_2 \
  --config config/data2_arch_config.json \
  --weights_out artifacts/weights_data2_2layer.json \
  --metrics_out artifacts/weights_data2_2layer_metrics.json

python3 evaluate.py \
  --test_dir datasets/data_2 \
  --config config/data2_arch_config.json \
  --weights artifacts/weights_data2_2layer_best.json
cd ..
```

## Achieved Results
From:
- `assignment1/artifacts/weights_data1_split_metrics.json`
- `assignment1/artifacts/weights_data2_2layer_metrics.json`

### Dataset A (`data_1`, 10 classes)
- Best epoch: `3`
- Best validation accuracy: `0.9602` (96.02%)
- Best test accuracy: `0.9628` (96.28%)
- Test loss (best): `0.1212`

### Dataset B (`data_2`, 100 classes)
- Best epoch: `7`
- Best validation accuracy: `0.3516` (35.16%)
- Best test accuracy: `0.3532` (35.32%)
- Test loss (best): `2.7037`

## Collaborators
Group 61: [G16]  
Amit Pandey [24m0792@iitb.ac.in]  
Sakshi Pandey [sakshipandey@iitb.ac.in]  
Sharath H N [24m2130@iitb.ac.in]
