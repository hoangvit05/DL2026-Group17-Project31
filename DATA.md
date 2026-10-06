# Dataset Documentation (DATA.md)

This document provides complete documentation for the clinical benchmark dataset used in the study:  
**"Medical Image Segmentation with Limited Training Data: Anatomical Lung Boundary Delineation on Chest X-Rays via Deep Learning"**.

---

## 🌐 1. Official Dataset Details

* **Dataset Name:** Chest X-Ray Masks and Labels
* **Official URL:** [https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
* **Dataset Version:** Version 2 (Published on Kaggle)
* **License:** CC0: Public Domain
* **Clinical Modality:** Posteroanterior (PA) 2D Chest Radiographs (CXR) and manual pixel-level binary annotations of both left and right lung fields.
* **Clinical Cohorts:**
  1. **Montgomery County CXR Set (USA):** Collected by the Montgomery County Department of Health and Human Services (Maryland, USA) in collaboration with the U.S. National Library of Medicine (NLM / NIH).
  2. **Shenzhen Hospital CXR Set (China):** Collected by Shenzhen No. 3 People's Hospital, Guangdong Medical College, China.
* **Total Valid Curated Pairs:** **704 pairs** with confirmed, verified ground truth masks:
  * Montgomery County: 138 pairs (138 raw CXR images + 138 paired binary masks).
  * Shenzhen Hospital: 566 pairs (566 raw CXR images + 566 paired binary masks).

---

## 📂 2. Directory Structure After Extraction

When downloading and extracting the official dataset, the directory structure must be organized as follows:

```text
Lung Segmentation/
├── CXR_png/
│   ├── CHNCXR_0001_0.png
│   ├── CHNCXR_0002_0.png
│   ├── ...
│   ├── MCUCXR_0001_0.png
│   └── MCUCXR_0002_0.png
└── masks/
    ├── CHNCXR_0001_0_mask.png
    ├── CHNCXR_0002_0_mask.png
    ├── ...
    ├── MCUCXR_0001_0.png
    └── MCUCXR_0002_0.png
```

> **Note on Naming Conventions:**
> * For the Shenzhen cohort (`CHNCXR_*`), CXR images end with `.png` and corresponding masks end with `_mask.png`.
> * For the Montgomery cohort (`MCUCXR_*`), CXR images and masks share the identical filename (`MCUCXR_*.png`).

---

## 🔀 3. Data Partitioning & Splits

To guarantee unbiased scientific comparison and strictly simulate clinical data scarcity regimes, the dataset is split using stratified random sampling with a fixed random seed (`seed = 42`):

### 3.1. Independent Test & Validation Sets
* **Fixed Independent Test Set (`test_fixed_20pct.csv`):** **$20\%$ (140 cases)**  
  Strictly held out and isolated across all experiments. No model ever trains or validates on this split.
* **Fixed Validation Set (`val_fixed.csv`):** **84 cases** (15% of the remaining training-validation pool)  
  Used exclusively for model checkpointing and tracking validation Dice loss.

### 3.2. Nested Training Subsets (Limited Data Regimes)
From the remaining training pool of **480 cases (100%)**, nested subsets are hierarchically sampled:

| Split Name | Ratio (%) | Number of Images | Clinical Data Regime |
| :--- | :---: | :---: | :--- |
| `train_10pct.csv` | **10%** | **48** | Constrained Annotation Scenario (Low-data regime) |
| `train_25pct.csv` | **25%** | **120** | Low-to-Moderate Data Regime |
| `train_50pct.csv` | **50%** | **240** | Moderate Data Regime |
| `train_100pct.csv` | **100%** | **480** | Full Supervised Baseline |

---

## 🔬 4. Preprocessing & Data Augmentation Pipeline

### 4.1. Preprocessing Procedure
1. **Image Loading & Channels:** Raw radiographs are read via OpenCV and converted from BGR to 3-channel RGB (`cv2.COLOR_BGR2RGB`) to ensure compatibility with ImageNet-pretrained feature encoders.
2. **Mask Binarization:** Ground truth masks are loaded in grayscale and thresholded:
   $$\text{Mask}_{\text{binary}} = (\text{Pixel Intensity} > 127) \rightarrow \{0.0, 1.0\}$$
3. **Spatial Normalization:** Bilinear spatial interpolation resizing both raw CXR images and ground truth masks to uniform dimensions of **$256 \times 256$ pixels**.
4. **Intensity Standardization:** Normalized per channel using ImageNet statistics:
   * Mean: $[0.485, 0.456, 0.406]$
   * Standard Deviation: $[0.229, 0.224, 0.225]$

### 4.2. Data Augmentation (Albumentations Pipeline)
To prevent severe overfitting under constrained data regimes ($10\%$ and $25\%$), the training pipeline applies:
* **Horizontal Flip:** Probability $p = 0.5$ (preserving anatomical symmetry).
* **Random Affine Transformations:** Scale variation $\in [0.9, 1.1]$, rotation $\in [-15^{\circ}, 15^{\circ}]$, translation $\in [\pm 6.25\%]$ with $p = 0.5$.
* **Photometric Perturbations:** Random brightness and contrast adjustment within $\pm 15\%$ ($p = 0.3$) to simulate scanner calibration variances.

---

## 💻 5. Scripts Required to Reproduce the Data

### 5.1. Automated Download via Kaggle CLI
Execute the following commands to download and unzip the dataset:

```bash
# Ensure Kaggle API token is configured (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN env var)
kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels --unzip
```

### 5.2. Automated Dataset Split Reproduction Script
You can reproduce the exact dataset partition CSV files anytime by running the following Python command:

```python
from src.dataset import prepare_splits

# Generate reproducible splits into the splits/ directory
prepare_splits(
    data_dir="Lung Segmentation",
    output_dir="splits",
    test_ratio=0.20,
    seed=42
)
```

Or execute directly from terminal:
```bash
python -c "from src.dataset import prepare_splits; prepare_splits()"
```

This generates all 6 split manifest CSV files inside `splits/`:
* `splits/test_fixed_20pct.csv`
* `splits/val_fixed.csv`
* `splits/train_10pct.csv`
* `splits/train_25pct.csv`
* `splits/train_50pct.csv`
* `splits/train_100pct.csv`
