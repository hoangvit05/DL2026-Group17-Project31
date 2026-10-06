# Training and Evaluation Instructions

This document provides comprehensive, step-by-step instructions to set up the environment, prepare the dataset, and execute both **Training** and **Evaluation** workflows for the benchmark study:  
**"Medical Image Segmentation with Limited Training Data: Anatomical Lung Boundary Delineation on Chest X-Rays via Deep Learning"**.

---

## 🛠️ 1. Environment & Dependencies Setup

### 1.1. Hardware Recommendations
* **Recommended Platform:** [Google Colab](https://colab.research.google.com/) with **GPU Tesla T4 (15GB VRAM)**.
* **Local Setup:** NVIDIA GPU with $\ge 8\text{GB}$ VRAM, CUDA 11.8+, and Python 3.10+.

### 1.2. Dependencies Installation
Install all required libraries specified in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

Core libraries include:
* `torch` & `torchvision`: Deep learning framework and tensor operations.
* `segmentation-models-pytorch`: Pretrained segmentation architectures (ResNet-34 backbone).
* `albumentations`: Medical image augmentation pipeline.
* `opencv-python`: Image I/O and spatial processing.
* `pandas` & `numpy`: Metadata manipulation and numerical metrics.
* `matplotlib` & `seaborn`: Visualization and publication-quality plotting.

---

## 📦 2. Dataset Preparation

### 2.1. Dataset Source
* **Official Kaggle Dataset:** [Chest X-Ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
* **Cohorts:** Montgomery County Set (138 pairs) + Shenzhen Hospital Set (566 pairs) = **704 verified pairs**.

### 2.2. Automated Download via Kaggle API
To download and extract the dataset automatically:

```bash
# Configure Kaggle credentials (or place kaggle.json in ~/.kaggle/)
kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels --unzip
```

### 2.3. Dataset Partitioning
The benchmark enforces a strict, reproducible data split:
* **Fixed Independent Test Set:** **$20\%$ (140 images)** held out for objective benchmarking.
* **Fixed Validation Set:** **84 images** for model checkpointing and early stopping.
* **Nested Training Subsets:** Progressively restricted to simulate data scarcity:
  * **5% Data:** 24 images
  * **10% Data:** 48 images
  * **25% Data:** 120 images
  * **50% Data:** 240 images
  * **100% Data:** 480 images (full training split)

---

## 🏋️ 3. Training Instructions

### 3.1. Model Architectures & Loss Function
1. **Conventional Baseline (From Scratch):**  
   * **`Vanilla U-Net`**: Standard 4-stage encoder-decoder architecture with Kaiming Normal weight initialization.
2. **Pretrained Model (Transfer Learning):**  
   * **`U-Net (ResNet-34)`**: Encoder initialized with ImageNet pretrained feature weights.
3. **Compound Loss Function:**
   $$\mathcal{L}_{\text{Combo}} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$$

### 3.2. Training Hyperparameters
| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **Optimizer** | `AdamW` | Fast convergence with decoupled weight decay |
| **Learning Rate** | `3e-4` | Stable fine-tuning for both scratch and pretrained backbones |
| **Weight Decay** | `1e-4` | L2 Regularization to mitigate overfitting in low-data regimes |
| **Scheduler** | `CosineAnnealingLR` | Smooth learning rate decay over $T_{\max} = 25$ |
| **Batch Size** | `8` | Fits within GPU memory while maintaining gradient variance |
| **Epochs** | `25` | Sufficient for full convergence across all data regimes |
| **Input Resolution** | `256 x 256` | Standard resolution balancing clinical detail and training throughput |

### 3.3. Training Execution

#### Method A: Command-Line Interface (CLI Scripts)
You can train specific model architectures under desired data subsets directly via terminal:

```bash
# Example 1: Train Pretrained U-Net (ResNet-34) on 10% data subset
python train.py --model unet --encoder resnet34 --ratio 10 --epochs 25 --batch_size 8 --lr 3e-4

# Example 2: Train Vanilla U-Net (from scratch) on 5% data subset
python train.py --model vanilla_unet --ratio 5 --epochs 25 --batch_size 8 --lr 3e-4

# Example 3: Train Pretrained U-Net on full 100% data
python train.py --model unet --encoder resnet34 --ratio 100 --epochs 25
```

Model checkpoints will be automatically saved to `checkpoints/{model}_{ratio}pct.pth`.

#### Method B: Google Colab Workflow
All 10 experimental configurations can also be executed in one click:
1. Open Google Colab and upload [`notebooks/lung_segmentation_colab.ipynb`](notebooks/lung_segmentation_colab.ipynb).
2. Ensure GPU is activated: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU** $\rightarrow$ **Save**.
3. Run Section 7 & 8 in the notebook to benchmark all subsets sequentially.

---

## 📈 4. Evaluation Instructions

### 4.1. Independent Test Benchmarking via CLI
To evaluate any trained model checkpoint on the isolated test set (140 images):

```bash
# Evaluate Pretrained U-Net (10% subset checkpoint)
python evaluate.py --checkpoint checkpoints/unet_10pct.pth --model unet --encoder resnet34

# Evaluate Vanilla U-Net (10% subset checkpoint)
python evaluate.py --checkpoint checkpoints/vanilla_unet_10pct.pth --model vanilla_unet
```

The script prints the quantitative **Dice Similarity Coefficient (DSC)** and **Intersection over Union (mIoU)** directly to stdout.

### 4.2. Metric Definitions & Benchmark Results
* **Dice Similarity Coefficient (DSC / F1-Score)**:
  $$\text{Dice} = \frac{2 \times |P \cap G|}{|P| + |G|}$$
* **Mean Intersection over Union (mIoU / Jaccard Index)**:
  $$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

All quantitative benchmark results across all 10 configurations are stored in:  
📁 [`data_scaling_benchmark_results.csv`](data_scaling_benchmark_results.csv)

### 4.3. Visual & Qualitative Generation
Run the visual reporting cells in the notebook to produce:
1. **`data_scaling_comparison.png`**: Multi-panel scaling curves comparing Dice and IoU across data regimes.
2. **`qualitative_comparison.png`**: High-resolution comparative matrix displaying raw images, ground truths, and segmented predictions across all models.

---

## 🎯 5. Artifact Checklist

Verify that the following output artifacts have been properly generated after execution:

- [x] `data_scaling_benchmark_results.csv` (Quantitative benchmark metrics)
- [x] `data_scaling_comparison.png` (Data scaling curve figures)
- [x] `qualitative_comparison.png` (Prediction visualization matrix)
- [x] `checkpoints/*.pth` (Saved best model weights)
