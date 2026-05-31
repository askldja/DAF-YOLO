# DAF-YOLO

Official implementation of **"Density-Aware Cross-Scale Feature Allocation for Lightweight UAV Small-Object Detection"**, submitted to *The Visual Computer*.

> **Note:** This repository contains the code corresponding to the above submission to *The Visual Computer*. If you use this code, please cite the paper (see [Citation](#citation)).

DAF-YOLO is a lightweight UAV small-object detector built on the YOLOv11n framework. It introduces a general, annotation-free **density-aware cross-scale feature allocation** mechanism, instantiated through three components:

- **DASSAFM** — Density-Aware Soft-Scale Adaptive Fusion Module: learns an implicit density-response map and uses it to guide per-location cross-scale feature allocation.
- **SDLKA** — Separable Dilated Large-Kernel Attention: enlarges the effective receptive field at low cost via directional dilated depthwise branches with channel–spatial co-selection.
- **LSDDetect** — Lightweight Cross-Scale Detection Head: shares global context across scales to improve cross-scale semantic consistency while controlling parameters.

On VisDrone2019, DAF-YOLO(n) reaches **39.0% mAP@0.5** and **23.1% mAP@0.5:0.95** with only **2.4M parameters**.

---

## Requirements

- Python 3.12
- PyTorch 2.7.0 (CUDA 12.8)
- NVIDIA GPU (experiments were run on a single RTX 4090D)

Install dependencies:

```bash
# 1. Create a clean environment (recommended)
conda create -n dafyolo python=3.12 -y
conda activate dafyolo

# 2. Install PyTorch matching your CUDA version
#    (the versions below were used in our experiments; see https://pytorch.org for other CUDA builds)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

---

## Datasets

Both datasets are publicly available.

- **VisDrone2019** — 10,209 images (6,471 train / 548 val / 3,190 test), 10 categories.
  Download: https://github.com/VisDrone/VisDrone-Dataset
- **TinyPerson** — used for cross-scene generalization.
  Download: https://github.com/ucas-vg/TinyBenchmark

After downloading, convert the annotations to YOLO format and organize the data as follows:
