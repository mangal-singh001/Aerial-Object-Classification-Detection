Here is your updated README with an additional closing line added at the end.
You can copy-paste this directly into GitHub 👇

---

# 🛩️ Aerial Object Classification & Detection

**🔍 Bird vs Drone Image Classification using Deep Learning**

This project involves building, evaluating, and deploying a deep-learning-based **binary image classifier** that identifies whether an aerial object is a **Bird** or a **Drone**.
A clean and efficient **Streamlit Web App (`app.py`)** is also included for real-time predictions.

---

## 📌 Project Overview

Ensuring airspace safety requires reliable systems to differentiate between drones and birds.
This project uses supervised deep learning to classify aerial object images into:

🕊️ **Bird**
🛸 **Drone**

The main notebook (`Aerial_Object.ipynb`) includes:
✔️ Data preprocessing
✔️ CNN/Transfer Learning model
✔️ Training, validation & testing
✔️ Evaluation metrics
✔️ Exporting the best-performing model

The Streamlit app (`app.py`) enables users to upload an image and get live predictions.

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

A dataset summary is available in `dataset_summary.csv`.

Typical folder structure:

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

This dataset contains balanced splits for training, validation, and testing.

---

## 🧠 Model Development (Notebook)

The notebook (`Aerial_Object.ipynb`) includes:

### ✔️ **Data Loading**

📥 Reading images
🖼️ Resizing images
⚙️ Normalizing pixel values
🔄 Converting images to tensors

### ✔️ **Data Augmentation**

🔁 Random flips
🔄 Rotation
🔍 Zoom
💡 Brightness adjustments

### ✔️ **Model Building**

Two approaches explored:
🧱 **Custom CNN**
🚀 **Transfer Learning** (ResNet, MobileNet, EfficientNet)

### ✔️ **Training Process**

⏳ Epochs
📦 Batch size
🛑 EarlyStopping
💾 ModelCheckpoint

### ✔️ **Model Evaluation**

📈 Accuracy
🎯 Precision
🔁 Recall
🏆 F1-Score
🔳 Confusion Matrix
📉 Training curves

### ✔️ **Model Export**

The trained model is saved as:

```
model.h5
```

Used later in the Streamlit app.

---

## 🚀 Streamlit Deployment

A lightweight UI created using Streamlit.

### ▶️ **How to run the Streamlit app**

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

### ✨ Features

📤 Upload image
🤖 Model predicts Bird or Drone
📊 Displays prediction confidence
⚡ Fast inference
🎨 Clean and simple UI

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mangal-singh001/Aerial-Object-Classification-Detection.git
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run notebook or deploy app

```bash
streamlit run app.py
```

---

## 🎥 Video Demonstration

If you want a quick walkthrough (8+ minutes):
📹 Explanation of project
🗂️ Dataset discussion
📘 Notebook walkthrough
📈 Model performance
🌐 Demo of Streamlit app

🎬 **Video Folder:**
[https://drive.google.com/drive/folders/1B1RQoYMZhbp3-3vKYlfSbT3xGW-g-owx?usp=sharing](https://drive.google.com/drive/folders/1B1RQoYMZhbp3-3vKYlfSbT3xGW-g-owx?usp=sharing)

---

## 📬 Contact

Feel free to reach out for suggestions or collaboration!

🔗 **GitHub:** [mangal-singh001](https://github.com/mangal-singh001)
🔗 **LinkedIn:** [Mangal Singh](https://www.linkedin.com/in/mangal-singh123/)

---

## ⭐ Final Note

If you find this project helpful, consider giving the repository a **star ⭐ on GitHub** — it motivates further improvements and new projects!

---

If you want, I can also add badges, GIF previews, or a results table.
