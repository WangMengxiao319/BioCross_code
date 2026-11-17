
# BioCross Repository

This repository contains the source code developed for the study:

**BioCross: A Cross-Modal Framework for Unified Representation of Multi-Modal Biosignals with Heterogeneous Metadata Fusion**

> **Publish**:Information Fusion 123 (2025) 103302. https://doi.org/10.1016/j.inffus.2025.103302.

## Overview
BioCross is a cross-modal framework designed to unify representations of multi-modal biosignals while incorporating heterogeneous metadata. It enables seamless integration of physiological signals such as ECG, PPG, and ABP with additional contextual metadata for enhanced diagnostic and predictive capabilities.
![framework.png](img%2Fframework.png)

### Features
- A mask-merge strategy to Variational Autoencoders architecture to align various modalities in a Gaussian latent space, which facilitates the effective representation of multi-sensor physiological data.

- Utilize the frequency-based attention mechanism and cross-attention to fuse embeddings of biosignals and metadata, includes circadian rhythms, enhancing the interaction of heterogeneous data.

- The product-of-experts technique to handle missing waveform inputs by enabling composable biosignal inputs. The output can either be certain modality generation or disease prediction via distinct decoders.

### Repository Structure
```plaintext
├── model/                 # Model architectures and configurations
├── utils/                 # Functions for data handling and model utilities
├── config/                # Config files
├── checkpoints/           # Model weights, organized by dataset and task
├── output/                # Generated waveforms, figures, metrics and statistics
├── data/                  # data folder
├── requirements.txt       # Python dependencies
├── Dt.py                  # Data processing script
├── S1_main_BioCross.py    # Main script for training and evaluation
├── S2_downstream_BioCross.py    # Downstream task training script
├── test_cross_modal.py    # Test script for cross-modal experiment
├── infer_cross_modal.py   # Single-sample inference and visualization
├── test_downstream.py     # Test script for downstream task
├── README.md              # Repository documentation (this file)
```

## Usage
For inference, load the [weights of BioCross](https://1drv.ms/f/c/664a0e01dd37c3c0/IgATHsfYHqD_QIh8HsGBdbbWAWaenhsQKntZbIezV6vH2ho?e=jDMJjG). This gives you a well-initialized model that can be fine-tuned for your own dataset or task.

Store downloaded weights separately from generated results:

```plaintext
checkpoints/
└── II_Pleth_ABP/
    ├── model.pth                 # Cross-modal BioCross model
    ├── heart_failure/model.pth   # Heart-failure classification model
    └── base/model.pth            # Optional base VAE weights for training (not bundled)
```

Run commands from the repository root. Single-sample and batch inference load
`checkpoints/II_Pleth_ABP/model.pth` by default. The `hwd` dataset uses the same
BioCross weights, with results saved under `output/hwd/single_sample/`.
Use `--checkpoint /path/to/model.pth` to override the single-sample model path.

```bash
python infer_cross_modal.py --task II_Pleth_ABP --sample-index 0 --target-modality both
python infer_cross_modal.py --task hwd --record 20261116_211644 --target-modality both --age 22 --gender 1
```

Training saves experiment weights under `checkpoints/` using the same relative
experiment directory as `output/`; figures and metrics stay under `output/`.
To deploy a trained cross-modal experiment, pass its `model.pth` using
`--checkpoint`, or copy it to the default `checkpoints/II_Pleth_ABP/model.pth`.
The training configuration's `encoder_pretrain_path` refers to the base VAE
checkpoint, which is separate from the full BioCross model.


## Mandatory Citation
If you use this code or dataset in your research, please cite the following paper:
```bibtex
@article{WANG2025103302,
title = {BioCross: A cross-modal framework for unified representation of multi-modal biosignals with heterogeneous metadata fusion},
journal = {Information Fusion},
volume = {123},
pages = {103302},
year = {2025},
doi = {https://doi.org/10.1016/j.inffus.2025.103302},
author = {Mengxiao Wang and Zhiyuan Li and Yuanyuan Tian and Xiaoyang Wei and Yanrui Jin and Chengliang Liu},
}
