# Medical Image Segmentation with Limited Training Data
## Đề tài: Phân vùng Phổi trên ảnh X-quang với lượng dữ liệu huấn luyện hạn chế (Lung Segmentation on Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dao-Trung-Hieu-2912/Deep-Learning/blob/main/notebooks/lung_segmentation_colab.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Dao-Trung-Hieu-2912/Deep-Learning)

> **GitHub Repository:** [https://github.com/Dao-Trung-Hieu-2912/Deep-Learning](https://github.com/Dao-Trung-Hieu-2912/Deep-Learning)  
> **Môn học:** Deep Learning (Học sâu)  
> **Lĩnh vực:** Computer Vision / Medical AI  
> **Đối tượng giải phẫu:** Hai lá phổi (Left Lung & Right Lung) trên ảnh X-quang lồng ngực (CXR)  
> **Môi trường thực thi chính:** **Google Colab (GPU Tesla T4 - 15GB VRAM)**  

---

## 📌 1. Giới thiệu tổng quan (Overview)

Trong chẩn đoán hình ảnh y tế, việc phân vùng chính xác ranh giới giải phẫu của hai lá phổi trên ảnh X-quang ngực (Chest X-Ray) là tiền đề bắt buộc cho các hệ thống CAD (Computer-Aided Diagnosis) tự động phát hiện lao phổi, viêm phổi, tràn dịch màng phổi hay COVID-19. Tuy nhiên, việc gán nhãn chi tiết từng pixel (pixel-level annotation) đòi hỏi bác sĩ chuyên khoa thực hiện, tốn nhiều chi phí và nhân lực.

Dự án này tập trung nghiên cứu, phát triển và đánh giá các giải pháp **Học sâu (Deep Learning)** trên môi trường **Google Colab** cho bài toán **Phân vùng cấu trúc giải phẫu Phổi** trong điều kiện **dữ liệu huấn luyện bị hạn chế (Limited Training Data)**.

### Mục tiêu khoa học:
1. **Khảo sát kiến trúc mô hình:** So sánh hiệu năng giữa mô hình phân vùng truyền thống huấn luyện từ đầu (*Conventional / Scratch*: Vanilla U-Net) với mô hình sử dụng bộ mã hóa tiền huấn luyện (*Pretrained Backbone*: U-Net + ResNet-34).
2. **Nghiên cứu suy giảm hiệu năng theo kích thước dữ liệu (Data Scaling Study):** Khảo sát đường cong hiệu năng khi giảm dần tập dữ liệu huấn luyện theo các mốc: **$5\%, 10\%, 25\%, 50\%, 100\%$**.
3. **Đánh giá sức bền của mô hình (Generalization & Robustness):** Khảo sát xem mô hình Pretrained duy trì độ chính xác và khả năng bảo toàn hình thái giải phẫu tốt hơn như thế nào khi lượng dữ liệu huấn luyện giảm sâu.

---

## 🩺 2. Bộ dữ liệu sử dụng (Dataset)

Dự án sử dụng bộ dữ liệu chuẩn y khoa **Chest X-Ray Masks and Labels (Montgomery County & Shenzhen Hospital)** được công bố bởi Viện Y tế Quốc gia Hoa Kỳ (NIH):

* **Nguồn dữ liệu:** [Kaggle - Chest X-Ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
* **Đối tượng:** Ảnh X-quang lồng ngực (Grayscale 2D) và mặt nạ nhãn nhị phân (Binary Mask) của hai lá phổi.
* **Tổng số mẫu có nhãn chuẩn:** **704 cặp ảnh - mặt nạ hợp lệ 100%**, bao gồm 2 nguồn bệnh viện:
  * **Tập Montgomery County (Mỹ):** 138 ca bệnh (138 ảnh X-quang + 138 mask phổi tương ứng).
  * **Tập Shenzhen Hospital (Trung Quốc):** 566 ca bệnh có đầy đủ mask phân vùng phổi.

---

## 🚀 3. Hướng dẫn chạy trọn gói trên Google Colab (1-Click Run)

Toàn bộ quy trình từ tải dữ liệu, tiền xử lý, huấn luyện 10 mô hình thực nghiệm và vẽ biểu đồ so sánh đã được đóng gói hoàn chỉnh trong **duy nhất 1 file Jupyter Notebook**:

📁 **[`notebooks/lung_segmentation_colab.ipynb`](notebooks/lung_segmentation_colab.ipynb)**

### Quy trình thực hiện trên Colab (1-Click Run):

File Notebook đã được **tích hợp sẵn API Token của bạn**, bạn **không cần upload thủ công bất kỳ file nào**:

1. **Mở Colab:** Truy cập [colab.research.google.com](https://colab.research.google.com/) $\rightarrow$ Chọn tab **Upload (Tải lên)** $\rightarrow$ Tải file [`notebooks/lung_segmentation_colab.ipynb`](notebooks/lung_segmentation_colab.ipynb) lên.
2. **Bật GPU:** Vào menu **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ Bấm **Save**.
3. **Bấm chạy toàn bộ:** Bấm menu **Runtime** $\rightarrow$ **Run all** (hoặc phím tắt `Ctrl + F9`).

Notebook sẽ tự động 100%:
* Xác thực API Kaggle bằng Token và tải dữ liệu siêu tốc (~50MB/s).
* Tự động cài thư viện và tạo các tập phân chia ($5\%, 10\%, 25\%, 50\%, 100\%$).
* Tự động huấn luyện chuỗi thí nghiệm đối chứng (Vanilla U-Net vs Pretrained U-Net).
* Tự động xuất bảng điểm tổng hợp và vẽ biểu đồ khoa học `data_scaling_comparison.png`.

---

## 🔬 4. Thiết kế thực nghiệm (Experimental Setup)

### 4.1. Phân chia dữ liệu (Data Splits)
* **Tập Test cố định (Fixed Test Set):** **$20\%$ (140 ảnh)** được giữ nguyên làm tập kiểm thử độc lập cho mọi kịch bản.
* **Tập Validation cố định:** **84 ảnh** theo dõi loss/dice trong quá trình học.
* **Tập Huấn luyện (Train Subsets):** Trích xuất mẫu theo tỷ lệ lồng nhau (nested sub-sampling):
  * **5% Data:** 24 ảnh (mô phỏng kịch bản cực ít dữ liệu - Few-shot).
  * **10% Data:** 48 ảnh.
  * **25% Data:** 120 ảnh.
  * **50% Data:** 240 ảnh.
  * **100% Data:** 480 ảnh (toàn bộ dữ liệu train).

### 4.2. Các mô hình đối chứng (Benchmark Models)
1. **Mô hình từ đầu (Conventional / Scratch):** `Vanilla U-Net` (kiến trúc tiêu chuẩn 4 tầng phân giải, khởi tạo trọng số ngẫu nhiên Kaiming Normal).
2. **Mô hình tiền huấn luyện (Transfer Learning):** `U-Net + ResNet-34 Encoder` (khởi tạo với trọng số ImageNet).

### 4.3. Hàm mất mát & Độ đo đánh giá
* **Hàm mất mát kết hợp:**
  $$\mathcal{L}_{\text{Combo}} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$$
* **Độ đo đánh giá chính:**
  * **Dice Similarity Coefficient (DSC / F1-Score)**
  * **Intersection over Union (mIoU / Jaccard Index)**

---

## 📊 5. Kết quả mong đợi trong báo cáo (Expected Deliverables)

1. **Bảng số liệu tổng hợp (Comparison Table):**
   * Bảng so sánh điểm Dice Score và IoU của cả 2 mô hình trên từng mốc $5\%, 10\%, 25\%, 50\%, 100\%$.
2. **Biểu đồ Data Scaling Curves (`data_scaling_comparison.png`):**
   * Thể hiện trực quan khoảng cách hiệu năng giữa *Pretrained U-Net* và *Vanilla U-Net* khi kích thước dữ liệu huấn luyện giảm dần.
3. **Trực quan hóa hình ảnh dự đoán:**
   * So sánh song song ảnh X-quang gốc, nhãn Ground Truth và vùng phân vùng do mô hình dự đoán.

---

## 📂 6. Cấu trúc thư mục dự án (Project Structure)

```text
Deep final/
├── Lung Segmentation/                      # Thư mục dữ liệu gốc (2.403 files nguyên vẹn)
│   ├── CXR_png/                            # Ảnh X-quang gốc (800 ảnh)
│   ├── masks/                              # Mặt nạ nhãn phổi (704 masks)
│   ├── test/                               # 96 ảnh test cuộc thi gốc
│   ├── ClinicalReadings/                   # Bệnh án lâm sàng bác sĩ
│   ├── NLM-ChinaCXRSet-ReadMe.docx         # Tài liệu NIH
│   └── NLM-MontgomeryCXRSet-ReadMe.pdf     # Tài liệu NIH
├── notebooks/
│   └── lung_segmentation_colab.ipynb        # NOTEBOOK CHÍNH: Chạy trọn gói trên Google Colab
├── chest-xray-masks-and-labels.zip         # File zip gốc tải từ Kaggle
└── README.md                               # Tài liệu hướng dẫn đồ án
```

---

## 👥 7. Thành viên nhóm & Phân công công việc (Team Members)

| STT | Họ và tên | Mã sinh viên | Nhiệm vụ chính |
| :-: | :--- | :---: | :--- |
| 1 | *Nguyễn Văn A* | *XXXXXXXX* | Chuẩn bị dữ liệu, chạy kịch bản Data Scaling trên Colab |
| 2 | *Trần Văn B* | *XXXXXXXX* | Xây dựng kiến trúc mô hình U-Net & Backbone ResNet-34 |
| 3 | *Lê Văn C* | *XXXXXXXX* | Đánh giá chỉ số Dice/IoU, tổng hợp biểu đồ và viết báo cáo |
