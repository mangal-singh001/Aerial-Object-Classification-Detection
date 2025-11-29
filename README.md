# 🛩️ Aerial Object Classification & Detection

**Bird vs Drone Image Classification using Deep Learning**

This project focuses on building, evaluating, and deploying a deep-learning-based **binary image classifier** that distinguishes **Birds** from **Drones** using aerial images.
A simple and effective **Streamlit Web App (`app.py`)** is also included for real-time image prediction.

---

## 📌 Project Overview

Airspace safety and wildlife monitoring require systems that can differentiate drones from birds. This project uses a supervised deep-learning model trained on aerial images belonging to two classes:

* **Bird**
* **Drone**

The notebook (`Aerial_Object.ipynb`) includes:
✔️ Data preprocessing
✔️ CNN/Transfer Learning model
✔️ Training, validation, and testing
✔️ Evaluation metrics
✔️ Saving the best-performing model

The Streamlit app (`app.py`) loads the trained model and allows users to upload any image for real-time prediction.

---

## 📂 Repository Structure

```
Aerial-Object-Classification-Detection/
│
├── Data/                     # Dataset folder (train/val/test)
│
├── Aerial_Object.ipynb       # Main Jupyter Notebook (model building & evaluation)
│
├── app.py                    # Streamlit application for deployment
│
├── dataset_summary.csv       # Dataset statistics (class counts, distribution)
│
├── Project Title.docx        # Original project brief / problem statement
│
├── requirements.txt          # Required libraries
│
└── README.md                 # Project documentation
```

---

## 📊 Dataset Summary

A short overview of the dataset is available in `dataset_summary.csv`.

Typical structure:

```
Data/
 ├── train/
 │    ├── bird/
 │    └── drone/
 ├── valid/
 │    ├── bird/
 │    └── drone/
 └── test/
      ├── bird/
      └── drone/
```

Dataset includes image counts for each class across train, validation, and test splits.

---

## 🧠 Model Development (Notebook)

The notebook (`Aerial_Object.ipynb`) covers:

### ✔️ **Data Loading**

* Reading images
* Resizing
* Normalizing
* Converting to tensors

### ✔️ **Data Augmentation**

* Random flips
* Rotation
* Zoom
* Brightness variation

### ✔️ **Model Building**

You may have used either:

* Custom CNN
  **or**
* Transfer Learning (ResNet / MobileNet / EfficientNet)

### ✔️ **Training**

* Epochs
* Batch size
* Callbacks (EarlyStopping, ModelCheckpoint)

### ✔️ **Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Loss/Accuracy plots

### ✔️ **Model Export**

The final model (`model.h5`) is saved for use in the Streamlit app.

---

## 🚀 Streamlit Deployment

A simple UI is implemented in `app.py`.

### **How to run the Streamlit app:**

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

### **Features:**

* Upload image (bird/drone)
* Model processes and displays prediction
* Shows classification confidence
* Clean UI and fast inference

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mangal-singh001/Aerial-Object-Classification-Detection.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebook or deploy Streamlit app

```bash
streamlit run app.py
```

---

## 🎥 Video Demonstration (Optional)

If you create an 8+ minute video:

* Intro about the project
* Dataset explanation
* Notebook walkthrough
* Model results
* Streamlit demo

https://drive.google.com/drive/folders/1B1RQoYMZhbp3-3vKYlfSbT3xGW-g-owx?usp=sharing

---

## 📬 Contact

For questions or suggestions:
**GitHub:** [mangal-singh001](https://github.com/mangal-singh001)

---
