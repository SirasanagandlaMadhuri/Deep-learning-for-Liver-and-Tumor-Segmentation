# Deep-learning-for-Liver-and-Tumor-Segmentation

This project focuses on accurate liver and tumor segmentation from abdominal CT scans using deep learning techniques. 
The models were trained and evaluated on the **Medical Segmentation Decathlon (MSD)** and **LiTS (Liver Tumor Segmentation)** datasets.
The project includes detailed experimentation, preprocessing, model training, inference, and a user-interactive web-based GUI.
---

## 🧠 Project Objective

To develop a deep learning-based segmentation system that can accurately delineate liver and tumor regions in CT scans. The model aims to assist in medical diagnostics by providing high-quality segmentation maps for clinical use.

---

## 🗂️ Datasets Used

### 1. Medical Segmentation Decathlon (MSD)
- Used grouped DICOM slices (64 per sample).
- Preprocessing involved intensity normalization and volume shaping.
- Trained the model for **133 epochs**.
- Achieved a **Dice Score of 0.93** for liver segmentation.
- However, due to off-normalization (`0 → background`, `>1 → liver`), tumors were not properly distinguished.
- Included:
  - `*.ipynb` training & preprocessing notebooks
  - `*.npy` datasets
  - Full set of project files and a presentation

### 2. LiTS (Liver Tumor Segmentation) Dataset
- Restarted from scratch due to limitations in MSD preprocessing.
- Preprocessed to preserve all class labels:
  - `0 → background`
  - `1 → liver`
  - `2 → tumor`
- Developed and trained the model successfully with tumor visibility.
- Performed inference with clear segmentation results.
- Included:
  - Jupyter notebooks
  - Working `.py` training and inference scripts
  - Saved model in `.pkl` format
  - Evaluation metrics saved as `.npy` files
  - GUI-based web interface for real-time image upload and prediction
  - Dataset images for both training and testing

---

## 🚀 Features

- ✅ High-accuracy segmentation model using U-Net/3D U-Net architectures
- ✅ Comprehensive training & inference pipelines
- ✅ Dice coefficient-based evaluation
- ✅ GUI web app for real-time segmentation using uploaded CT slices
- ✅ Complete documentation and research paper included

During experimentation, the MSD dataset was trained using a standard U-Net architecture, which provided good liver segmentation but failed to preserve tumor visibility due to preprocessing. In contrast, for the LiTS dataset, we used a more robust U-Net model with a ResNet-34 encoder as the backbone, which significantly improved both liver and tumor segmentation accuracy by leveraging deeper feature extraction capabilities.
## 🚀 GUI Application

A cross-platform **desktop GUI** is provided, allowing users to:
- Upload RGB `.jpg` slices of CT scans.
- Automatically run inference and get segmentation output.
- View original, grayscale, and predicted mask images side by side.
- Access contact information of contributors through clickable buttons.

### 🎨 GUI Highlights
- Built using **PyQt5**.
- Integrates **Matplotlib** for visual output.
- Background-threaded prediction using `QThread`.
- Loads the model once at startup (`sdp_best_seg_model.pkl`).

## 🧪 Results

| Dataset | Dice Score (Liver) | Tumor Visible | Notes |
|--------|---------------------|---------------|-------|
| MSD    | 0.93                | ❌           | Tumor info lost due to normalization |
| LiTS   | 0.9820              | ✅           | Accurate tumor segmentation |

- GUI outputs are visually validated against ground truth.
- Tumors clearly segmented in LiTS-based model.
- Real-time inference in GUI is fast and responsive.

---

## 🧰 Technologies Used

| Component      | Technology         |
|----------------|--------------------|
| Programming    | Python 3.x          |
| Deep Learning  | Fastai, PyTorch     |
| GUI            | PyQt5               |
| Image Handling | NumPy, PIL, OpenCV  |
| Visualization  | Matplotlib          |
| File I/O       | pathlib, QFileDialog|
| Model Format   | `.pkl` (Fastai model) |

## 📌 License

This repository is released for academic and research purposes.

## 📬 Contact

For any questions or collaborations:  
📧 *madhurisirasanagandla@gmail.com*

