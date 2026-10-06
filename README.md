# Medical Image Segmentation with Limited Training Data
## Benchmark Case Study: Anatomical Lung Boundary Delineation on Chest X-Rays via Deep Learning

> **Course:** Deep Learning  
> **Domain:** Computer Vision / Medical AI  
> **Benchmark Dataset:** Chest X-Rays (Montgomery County & Shenzhen Hospital Datasets)  
> **Execution Platform:** PyTorch on GPU (NVIDIA CUDA / RTX Acceleration)  

---

## 📌 1. Overview

In medical image analysis, accurate anatomical segmentation of the lung boundaries on Chest X-Rays (CXR) is an indispensable prerequisite for automated Computer-Aided Diagnosis (CAD) systems—facilitating timely detection of tuberculosis, pneumonia, pleural effusion, and COVID-19. However, annotating medical images at the pixel level requires specialized clinical expertise, incurring substantial financial costs, diagnostic delays, and specialized human labor.

This project investigates, develops, and evaluates **Deep Learning** solutions for **Anatomical Lung Segmentation** under **Limited Training Data** regimes.

### Scientific Objectives:
1. **Model Architecture Exploration:** Benchmark segmentation performance between a conventional model trained from scratch (*Conventional / Scratch*: Vanilla U-Net) and a transfer-learning model equipped with a pretrained feature encoder (*Pretrained Backbone*: U-Net + ResNet-34).
2. **Data Scaling Study:** Quantify performance degradation trajectories as training sample size is progressively restricted across defined proportions: **$10\%, 25\%, 50\%, 100\%$**.
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

👉 **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)** (Contains full instructions for hardware setup, dependencies, automated one-click benchmarking, CLI training loops, and evaluation metrics).

---

## 🔬 4. Experimental Setup

### 4.1. Data Splits
* **Fixed Independent Test Set:** **$20\%$ (140 images)** strictly held out for unbiased, objective evaluation across all configurations.
* **Fixed Validation Set:** **84 images** used for monitoring convergence and early checkpoint saving.
* **Nested Training Subsets:** Hierarchically sampled to emulate realistic data scarcity scenarios:
  * **10% Data:** 48 images (constrained annotation regime).
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
| **10% (48 images - Constrained)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **95.53%**<br>95.17% | **91.52%**<br>90.87% | **+0.36% Dice** \| **+0.65% IoU** |
| **25% (120 images)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **95.90%**<br>95.53% | **92.23%**<br>91.54% | **+0.37% Dice** \| **+0.69% IoU** |
| **50% (240 images)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **96.19%**<br>96.12% | **92.79%**<br>92.63% | **+0.07% Dice** \| **+0.16% IoU** |
| **100% (480 images - Full)** | **Pretrained U-Net (ResNet-34)**<br>Vanilla U-Net (From Scratch) | **96.39%**<br>96.30% | **93.13%**<br>92.95% | **+0.09% Dice** \| **+0.18% IoU** |

### 5.2. Data Scaling Performance Curves
 
![Data Scaling Curves: Training Data Ratio vs Test Dice Score and Test IoU](data_scaling_comparison.png)

### 5.3. Qualitative Segmentation Predictions

![Qualitative segmentation comparison across models and data scales](qualitative_comparison.png)

### 5.4. Key Scientific Findings
1. **Decisive Transfer Learning Advantage Under Data Scarcity:** When training data drops to $10\%$ (48 images), Pretrained U-Net consistently outperforms Vanilla U-Net (+**0.36%** Dice, +**0.65%** IoU) and sustains this advantage at $25\%$ (+**0.37%** Dice, +**0.69%** IoU).
2. **Clinical Annotation Burden Reduction:** Pretrained U-Net reaches **95.53%** Dice with merely **48 images (10%)**, matching Vanilla U-Net trained on **120 images (25% - 95.53%)**. This demonstrates that transfer learning reduces manual radiologist annotation costs by **60%** while delivering equivalent boundary quality.
3. **Anatomical Boundary Integrity:** Visual inspections confirm that Pretrained U-Net produces smooth, anatomically sound segmentations that preserve sharp delineation at the costophrenic angles, whereas Vanilla U-Net exhibits slight boundary blur under constrained data regimes.

---

## 📂 6. Project Structure

```text
Medical-Image-Segmentation-with-Limited-Training-Data/
├── src/
│   ├── dataset.py                          # Dataset loaders (Standard & Fast Cached), transforms, splits
│   ├── models.py                           # Vanilla U-Net and Pretrained ResNet-34 U-Net
│   ├── metrics.py                          # Compound loss (BCE + Dice) and evaluation metrics
│   └── __init__.py                         # Package initialization
├── splits/                                 # Stratified dataset split manifests (*.csv)
├── checkpoints/                            # Saved trained model weights (*.pth)
├── run_all.py                              # End-to-end automated 1-click benchmark runner
├── train.py                                # Modular standalone CLI training script
├── evaluate.py                             # Modular standalone CLI evaluation script
├── data_scaling_benchmark_results.csv       # Experimental benchmark metrics (Dice & IoU)
├── data_scaling_comparison.png             # Scientific data scaling curve visualization
├── qualitative_comparison.png              # Multi-patient qualitative prediction matrix
├── requirements.txt                        # Python dependencies and package specifications
├── INSTRUCTIONS.md                         # Detailed training and evaluation execution instructions
├── DATA.md                                 # Official dataset documentation and reproduction guide
├── .gitignore                              # Git exclusion rules for large datasets and caches
└── README.md                               # Project documentation and comprehensive benchmark report
```

---

## 👥 7. Phân công công việc (Team Members & Task Assignment)

Bảng phân công chi tiết vai trò, trách nhiệm và các mô-đun/tập tin do 7 thành viên trong nhóm đảm nhận:

| STT | Họ và tên | Vai trò (Role) | Mô-đun & File đảm nhận | Nhiệm vụ chính & Đóng góp |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Đặng Việt Hoàng** *(Leader)* | Trưởng nhóm / Kiến trúc hệ thống | `.gitignore`, Quản trị kho lưu trữ, Setup Google Colab ban đầu | Lập kế hoạch dự án, phân chia công việc, khởi tạo Git repository, xây dựng notebook thực nghiệm mẫu ban đầu trên Google Colab và rà soát tiến độ nhóm. |
| 2 | **Đào Trung Hiếu** | Kỹ sư Học sâu (Core Deep Learning) | `src/models.py`, `train.py`, `run_all.py`, Checkpoints logic | Xây dựng kiến trúc Vanilla U-Net & U-Net + ResNet-34 Encoder, thiết kế hàm mất mát tích hợp Hybrid Loss (BCE + Dice), xây dựng pipeline tự động hóa huấn luyện đầu-cuối (`run_all.py`), cơ chế lưu checkpoint mô hình. |
| 3 | **Nguyễn Quang Thuần** | Kỹ sư Dữ liệu (Data Pipeline) | `src/dataset.py`, `DATA.md`, `splits/` (`splits/*.csv`) | Thu thập và tiền xử lý bộ dữ liệu Chest X-Ray (Montgomery & Shenzhen), xây dựng DataLoader kèm Fast In-Memory Caching, phân chia tập dữ liệu phân tầng ($10\%, 25\%, 50\%, 100\%$ và test cố định $20\%$), biên soạn `DATA.md`. |
| 4 | **Nguyễn Đăng Hoàng** | Kỹ sư Đánh giá & Kiểm thử (Evaluation) | `src/metrics.py`, `evaluate.py` | Xây dựng mô-đun tính toán các chỉ số kiểm thử y tế (Dice Similarity Coefficient - DSC, mIoU / Jaccard Index), lập trình CLI đánh giá độc lập (`evaluate.py`), chuẩn hóa mã nguồn đánh giá. |
| 5 | **Nguyễn Minh Đức** | Chuyên viên Thực nghiệm & Đo điểm chuẩn | `data_scaling_benchmark_results.csv`, Bảng số liệu benchmark | Thiết kế và thực thi kịch bản đo điểm chuẩn Data Scaling qua 4 mốc dữ liệu ($10\%, 25\%, 50\%, 100\%$), đo lường độ suy giảm hiệu năng, tổng hợp kết quả định lượng vào bảng dữ liệu so sánh. |
| 6 | **Nguyễn Tú Oanh** | Phân tích Định tính & Trực quan hóa | `qualitative_comparison.png`, `data_scaling_comparison.png` | Xử lý hậu kỳ trực quan hóa phân vùng đa bệnh nhân (`qualitative_comparison.png`), phân tích định tính sự bảo toàn giải phẫu góc sườn hoành (costophrenic angles), vẽ biểu đồ đường cong Data Scaling (`data_scaling_comparison.png`). |
| 7 | **Đào Minh Nguyệt** | Soạn thảo Tài liệu & Báo cáo Kỹ thuật | `README.md`, `INSTRUCTIONS.md`, `requirements.txt` | Soạn thảo và chuẩn hóa toàn bộ tài liệu kỹ thuật, hướng dẫn cài đặt và chạy thực nghiệm (`INSTRUCTIONS.md`), quản lý phụ thuộc thư viện (`requirements.txt`), biên dịch tiếng Anh học thuật cho báo cáo dự án. |
