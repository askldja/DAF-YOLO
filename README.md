# DAF-YOLO

Official implementation of **"Density-Guided Cross-Scale Feature Allocation for UAV Small-Object Detection"**, submitted to *Multimedia Systems*.

> **Note:** This repository contains the implementation associated with our manuscript submitted to *Multimedia Systems*. If you find this work useful, please cite our paper (see [Citation](#citation)).

DAF-YOLO is a parameter-compact UAV small-object detection framework developed upon YOLOv11n. It introduces a density-guided cross-scale feature allocation mechanism through three complementary components:

* **DASSAFM** — Density-Aware Soft-Scale Adaptive Fusion Module: learns an implicit density-response map without requiring additional density annotations and uses it to modulate spatially adaptive cross-scale fusion weights.
* **SDLKA** — Separable Dilated Large-Kernel Attention: enhances directional contextual representation through separable dilated depthwise convolutions and stage-specific channel–spatial co-selection.
* **LSDDetect** — Lightweight Cross-Scale Detection Head: aggregates and redistributes semantic context across detection scales while maintaining a compact parameter budget.

On the VisDrone2019 validation set, **DAF-YOLO(n) achieves 39.0% mAP@0.5 and 23.1% mAP@0.5:0.95 with 2.4M parameters and 13.3 GFLOPs**.

Although DAF-YOLO reduces model parameters compared with the YOLOv11n baseline, high-resolution cross-scale fusion introduces additional computational overhead. Therefore, the proposed method is characterized as parameter-compact rather than strictly computation-lightweight.


## Overview

DAF-YOLO is a **parameter-compact** object detector for small objects in UAV imagery, developed from the YOLO11n (YOLOv11n) baseline. Its core idea is to adapt cross-scale feature allocation to local object-density patterns through a learned density response, **without requiring additional density annotations** beyond the object-detection labels.

The framework contains three components:

- **DASSAFM (Density-Aware Soft-Scale Adaptive Fusion Module):** Learns an implicit density-response map from the detection objective and uses it to modulate spatially varying scale-selection logits for cross-scale feature fusion.
- **SDLKA (Separable Dilated Large-Kernel Attention):** Combines local and directional dilated depthwise operations to capture contextual information with a limited parameter budget. A stage-specific C3k2-SDLKA design adjusts the auxiliary operations across backbone levels.
- **LSDDetect (Lightweight Cross-Scale Detection Head):** Aggregates shared semantic context across three detection scales before lightweight prediction, with the aim of improving cross-scale semantic consistency.

## Results

On the **VisDrone2019-DET validation split**, DAF-YOLO(n) achieves **39.0% mAP@0.5** and **23.1% mAP@0.5:0.95** with approximately **2.4 million parameters**.

| Model | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Parameters (M) | GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| YOLOv11n | 32.8 | 18.9 | 2.6 | 6.5 |
| DAF-YOLO(n) | **39.0** | **23.1** | **2.4** | 13.3 |
| DAF-YOLO(s) | 42.8 | 25.8 | 7.4 | 33.0 |

Results above are from author-conducted experiments on VisDrone2019 validation data, using **640 × 640** input images and **300 training epochs**, with models trained **from scratch (no pretrained weights)**.

**Efficiency note.** DAF-YOLO(n) reduces the parameter count relative to YOLOv11n, but its high-resolution fusion and context modeling increase computation (6.5 → 13.3 GFLOPs). In the reported GPU-based evaluation, inference latency increases from 5.4 ms to 6.9 ms and throughput decreases from 185 to 145 FPS. Accordingly, DAF-YOLO is described as *parameter-compact*, not strictly computation-lightweight. These results should not be interpreted as verified on-board UAV performance; evaluation on embedded UAV hardware remains future work.

## Requirements

The reported experimental environment was:

- Python 3.12
- PyTorch 2.7.0 and torchvision 0.22.0 (CUDA 12.8 build)
- NVIDIA RTX 4090D GPU

Create the environment and install dependencies:

```bash
conda create -n dafyolo python=3.12 -y
conda activate dafyolo

# Install the PyTorch build matching your CUDA environment.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install repository dependencies.
pip install -r requirements.txt
```

For alternative operating systems and CUDA versions, consult the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Datasets

The experiments use two publicly available datasets:

- **VisDrone2019-DET:** 10,209 images in total (6,471 training, 548 validation, and 3,190 testing images) across 10 object categories. [Dataset repository](https://github.com/VisDrone/VisDrone-Dataset).
- **TinyPerson:** Used to investigate cross-scene generalization. [TinyBenchmark repository](https://github.com/ucas-vg/TinyBenchmark).

Download the datasets and convert their annotations to the YOLO detection format. Each label file should contain one line per object in the form:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates must be normalized by image width and height. Ensure category IDs and dataset splits match the configuration files used for each experiment. A typical directory layout is:

```text
datasets/
├── VisDrone2019/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       └── val/
└── TinyPerson/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

> This layout is illustrative. Set the dataset paths in your YAML configuration files to the actual locations. The annotation-conversion scripts and exact data configuration filenames should be documented according to the files included in this repository.

## Architecture Details

For SDLKA, the adopted stage-specific configuration uses:

| Backbone stage | Variant | Large-kernel branch | Channel–spatial co-selection (CSCS) | MLP expansion ratio |
| --- | --- | --- | --- | --- |
| C2 | L | Enabled | Enabled | 2.0 |
| C3 | M | Enabled | Disabled | 1.0 |
| C4 | S | Enabled | Disabled | Not used |
| C5 | S | Enabled | Disabled | Not used |

The reported SDLKA directional branch uses kernel size **7** and dilation rate **2**; the shared-context feature width in LSDDetect is **96 channels**. These are the adopted settings among the evaluated configurations, not claims of global optimality.

## Training and Evaluation

The reported VisDrone2019 comparison uses the following experimental protocol:

| Setting | Value |
| --- | --- |
| Dataset split for evaluation | VisDrone2019-DET validation (548 images) |
| Input size | 640 × 640 |
| Training epochs | 300 |
| Initialization | From scratch, without pretrained weights |
| Hardware used in experiments | Single NVIDIA RTX 4090D |

**Repository-specific commands:** Add the exact training, validation, inference, and annotation-conversion commands here after confirming the entry-point scripts, model YAML files, dataset YAML files, and checkpoint paths in the repository. Unverified commands have intentionally not been provided.

## Limitations

Although DAF-YOLO(n) uses fewer parameters than YOLOv11n, its high-resolution cross-scale fusion and contextual enhancement incur additional FLOPs and inference latency. The reported speed measurements are GPU-based; on-board deployment on resource-constrained UAV platforms has not yet been validated. Further work may explore pruning, quantization, and hardware-specific inference acceleration.

## Citation

The manuscript is under submission. Replace the following placeholder with the final bibliographic entry when available:

```bibtex
@article{dafyolo,
  title   = {Density-Aware Cross-Scale Feature Allocation for Lightweight UAV Small-Object Detection},
  author  = {To be updated},
  journal = {The Visual Computer},
  note    = {Manuscript under submission}
}
```

## Acknowledgments

DAF-YOLO is developed from the YOLO11/YOLOv11 family. Please also acknowledge the original frameworks and dataset authors where appropriate.
