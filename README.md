
# BioCross Repository

This repository contains the source code developed for the study:

**BioCross: A Cross-Modal Framework for Unified Representation of Multi-Modal Biosignals with Heterogeneous Metadata Fusion**

> **Status:** Currently under pre-submission review.

## Overview
BioCross is a cross-modal framework designed to unify representations of multi-modal biosignals while incorporating heterogeneous metadata. It enables seamless integration of physiological signals such as ECG, PPG, and ABP with additional contextual metadata for enhanced diagnostic and predictive capabilities.
![framework.png](img%2Fframework.png)

## Features
- A mask-merge strategy to Variational Autoencoders architecture to align various modalities in a Gaussian latent space, which facilitates the effective representation of multi-sensor physiological data.

- Utilize the frequency-based attention mechanism and cross-attention to fuse embeddings of biosignals and metadata, includes circadian rhythms, enhancing the interaction of heterogeneous data.

- The product-of-experts technique to handle missing waveform inputs by enabling composable biosignal inputs. The output can either be certain modality generation or disease prediction via distinct decoders.

## Repository Structure
```plaintext
├── model/                # Model architectures and configurations
├── utils/                 # Helper functions for data handling and model utilities
├── requirements.txt       # Python dependencies
├── Dt.py                  # Data processing script
├── S1_main_BioCross.py    # Main script for training and evaluation
├── S2_downstream_BioCross.py    # Downstream task training script
├── README.md              # Repository documentation (this file)
```



## Contact
For questions or collaborations, please contact:

- **Name:** Mengxiao Wang
- **Email:** mengxiaowang@sjtu.edu.cn
