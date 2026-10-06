# Medical Image Segmentation with Limited Training Data
## Benchmark Case Study: Anatomical Lung Boundary Delineation on Chest X-Rays via Deep Learning (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoangvit05/Medical-Image-Segmentation-with-Limited-Training-Data/blob/main/notebooks/lung_segmentation_colab.ipynb)

> **Course:** Deep Learning  
> **Domain:** Computer Vision / Medical AI  
> **Benchmark Dataset:** Chest X-Rays (Montgomery County & Shenzhen Hospital Datasets)  
> **Primary Execution Environment:** **Google Colab (GPU Tesla T4 - 15GB VRAM)**  

---

## 📌 1. Overview

In medical image analysis, accurate anatomical segmentation of the lung boundaries on Chest X-Rays (CXR) is an indispensable prerequisite for automated Computer-Aided Diagnosis (CAD) systems—facilitating timely detection of tuberculosis, pneumonia, pleural effusion, and COVID-19. However, annotating medical images at the pixel level requires specialized clinical expertise, incurring substantial financial costs, diagnostic delays, and specialized human labor.

This project investigates, develops, and evaluates **Deep Learning** solutions on **Google Colab** for **Anatomical Lung Segmentation** under **Limited Training Data** regimes.

### Scientific Objectives:
1. **Model Architecture Exploration:** Benchmark segmentation performance between a conventional model trained from scratch (*Conventional / Scratch*: Vanilla U-Net) and a transfer-learning model equipped with a pretrained feature encoder (*Pretrained Backbone*: U-Net + ResNet-34).
2. **Data Scaling Study:** Quantify performance degradation trajectories as training sample size is progressively restricted across defined proportions: **$5\%, 10\%, 25\%, 50\%, 100\%$**.
3. **Generalization & Robustness Evaluation:** Compare generalization capability, accuracy retention, and anatomical shape preservation between the conventional model (Conventional Vanilla U-Net) and the pretrained model (Pretrained U-Net) under severe data scarcity.

---

## 🩺 2. Dataset

The project employs the clinical benchmark dataset **Chest X-Ray Masks and Labels (Montgomery County & Shenzhen Hospital)** released by the U.S. National Institutes of Health (NIH):

* **Source:** [Kaggle - Chest X-Ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
* **Target:** 2D Grayscale Chest X-Rays and corresponding binary ground-truth segmentation masks of both lungs.
* **Total Valid Annotated Pairs:** **704 verified pairs**, curated from two distinct hospital cohorts:
  * **Montgomery County Set (USA):** 138 cases (138 CXR images + 138 paired lung masks).
  * **Shenzhen Hospital Set (China):** 566 cases with complete lung masks.

👉 Detailed dataset specifications, download links, stratified splits, and data reproduction instructions are fully documented in:  
**[`DATA.md`](DATA.md)**

---

## 🚀 3. Training & Evaluation Guide

Detailed instructions on environment setup, dataset downloading, model training, and evaluation are documented in:

👉 **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)** (Contains full instructions for hardware setup, dependencies, training loops, evaluation metrics, and artifact reproduction).

---

## 🔬 4. Experimental Setup

### 4.1. Data Splits
* **Fixed Independent Test Set:** **$20\%$ (140 images)** strictly held out for unbiased, objective evaluation across all configurations.
* **Fixed Validation Set:** **84 images** used for monitoring convergence and early checkpoint saving.
* **Nested Training Subsets:** Hierarchically sampled to emulate realistic data scarcity scenarios:
  * **5% Data:** 24 images (extreme few-shot scenario).
  * **10% Data:** 48 images (constrained regime).
  * **25% Data:** 120 images (low-to-moderate data).
  * **50% Data:** 240 images (moderate data).
  * **100% Data:** 480 images (full training split).

### 4.2. Benchmark Models
1. **Conventional Model (From Scratch):** `Vanilla U-Net` (standard 4-stage resolution encoder-decoder with Kaiming Normal weight initialization).
2. **Pretrained Model (Transfer Learning):** `U-Net + ResNet-34 Encoder` (encoder initialized with ImageNet weights).

### 4.3. Loss Function & Evaluation Metrics
* **Compound Hybrid Loss:**
  $$\mathcal{L}_{\text{Combo}} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$$
* **Primary Evaluation Metrics:**
  * **Dice Similarity Coefficient (DSC / F1-Score)**
  * **Intersection over Union (mIoU / Jaccard Index)**

---

## 📊 5. Experimental Results & Scientific Analysis

All evaluations were conducted on the identical, independent test set of **140 images (20% data)** and a fixed validation set of **84 images**.

### 5.1. Benchmark Quantitative Results

| Training Data Scale (Images) | Model Architecture | Test Dice Score (%) | Test IoU (%) | Delta (Pretrained vs Scratch) |
| :--- | :--- | :---: | :---: | :---: |
| **5% (24 images - Few-shot)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **94.88%**<br>94.23% | **90.34%**<br>89.19% | **+0.65% Dice** \| **+1.15% IoU** |
| **10% (48 images)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **95.74%**<br>94.80% | **91.92%**<br>90.22% | **+0.94% Dice** \| **+1.70% IoU** |
| **25% (120 images)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **96.19%**<br>95.64% | **92.77%**<br>91.74% | **+0.55% Dice** \| **+1.03% IoU** |
| **50% (240 images)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **96.45%**<br>96.13% | **93.26%**<br>92.66% | **+0.32% Dice** \| **+0.60% IoU** |
| **100% (480 images)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | 96.44%<br>**96.49%** | 93.24%<br>**93.33%** | Near Parity Saturation (~96.5%) |

### 5.2. Data Scaling Performance Curves
 
![Data Scaling Curves: Training Data Ratio vs Test Dice Score and Test IoU](data_scaling_comparison.png)

### 5.3. Qualitative Segmentation Predictions

![Qualitative segmentation comparison across models and data scales](qualitative_comparison.png)

### 5.4. Key Scientific Findings
1. **Decisive Transfer Learning Advantage Under Data Scarcity:** When training data drops to $10\%$ (48 images), Pretrained U-Net significantly outperforms Vanilla U-Net (+**0.94%** Dice, +**1.70%** IoU), and maintains a clear lead at the extreme $5\%$ data scale (+**0.65%** Dice, +**1.15%** IoU).
2. **Clinical Annotation Burden Reduction:** Pretrained U-Net reaches **95.74%** Dice with merely **48 images (10%)**, outperforming Vanilla U-Net trained on **120 images (25% - 95.64%)**. This confirms that Transfer Learning cuts radiologist annotation requirements by more than half while sustaining superior accuracy.
3. **Anatomical Boundary Integrity:** Visual inspections reveal that under extreme data constraints (5% data, 24 images), Vanilla U-Net generates jagged boundaries and boundary dropouts at the costophrenic angles. In contrast, Pretrained U-Net maintains smooth, anatomically faithful contours across both normal lungs and complex pathological lesions (such as tuberculosis and pleurisy).

---

## 📂 6. Project Structure

```text
Medical-Image-Segmentation-with-Limited-Training-Data/
├── src/
│   ├── dataset.py                          # Dataset loader, transforms, and split generator
│   ├── models.py                           # Vanilla U-Net and Pretrained ResNet-34 U-Net
│   ├── metrics.py                          # Compound loss (BCE + Dice) and evaluation metrics
│   └── __init__.py                         # Package initialization
├── notebooks/
│   └── lung_segmentation_colab.ipynb       # Main executable Google Colab notebook (with outputs)
├── splits/                                 # Stratified dataset split manifests (*.csv)
├── train.py                                # Standalone CLI training script
├── evaluate.py                             # Standalone CLI evaluation script
├── data_scaling_benchmark_results.csv       # Experimental benchmark metrics (Dice & IoU)
├── data_scaling_comparison.png             # Scientific data scaling curve visualization
├── qualitative_comparison.png              # Multi-patient qualitative prediction matrix
├── requirements.txt                        # Python dependencies and package specifications
├── INSTRUCTIONS.md                         # Detailed training and evaluation execution instructions
├── DATA.md                                 # Official dataset documentation and reproduction guide
├── .gitignore                              # Git exclusion rules for large datasets and caches
└── README.md                               # Project documentation and comprehensive benchmark report
```
