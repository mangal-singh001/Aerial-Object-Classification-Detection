# Aerial Object Classification & Detection

**Binary classification (Bird vs Drone) + optional YOLOv8 object detection**
A compact end-to-end project for building, evaluating, and deploying deep-learning models that distinguish birds from drones in aerial imagery and — optionally — detect and localize them in frames. Project overview and dataset specs adapted from the project brief. 

---

## Table of contents

1. [Project Overview](#project-overview)
2. [Key Features & Skills](#key-features--skills)
3. [Repository structure](#repository-structure)
4. [Datasets & format](#datasets--format)
5. [Quickstart — Installation](#quickstart--installation)
6. [How to run (examples)](#how-to-run-examples)
7. [Model training & evaluation](#model-training--evaluation)
8. [YOLOv8 (optional) — object detection](#yolov8-optional---object-detection)
9. [Streamlit deployment](#streamlit-deployment)
10. [Results & deliverables](#results--deliverables)
11. [Tips & troubleshooting](#tips--troubleshooting)
12. [Contributing](#contributing)
13. [License & Contact](#license--contact)

---

## Project overview

The goal is to build an accurate image classification model that separates **Bird** vs **Drone** from aerial images and (optionally) a YOLOv8-based object detector to localize these objects in frames for real-time/near-real-time use cases such as wildlife monitoring, airport safety, and airspace security. Project details (problem statement, workflow, dataset counts, and deliverables) are summarized from the project document. 

---

## Key features & skills

* Binary image classification (custom CNN + transfer learning: ResNet50/MobileNet/EfficientNet)
* Data preprocessing and augmentation
* Model evaluation: accuracy, precision, recall, F1-score, confusion matrix
* Optional object detection with YOLOv8 (annotation format, training, inference)
* Streamlit web app for interactive deployment
* Deliverables: notebooks/scripts, trained model files, Streamlit app

---

## Repository structure (recommended)

```
Aerial-Object-Classification-Detection/
├─ data/
│  ├─ classification_dataset/
│  │  ├─ train/
│  │  │  ├─ bird/
│  │  │  └─ drone/
│  │  ├─ valid/
│  │  └─ test/
│  └─ object_detection_Dataset/    # YOLOv8 format: images + .txt per image
├─ notebooks/
│  ├─ 01_explore_dataset.ipynb
│  ├─ 02_preprocessing_augmentation.ipynb
│  ├─ 03_train_classification.ipynb
│  ├─ 04_evaluate_models.ipynb
│  └─ 05_yolov8_detection.ipynb
├─ src/
│  ├─ datasets.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ utils.py
├─ streamlit_app/
│  ├─ app.py
│  └─ requirements.txt
├─ models/                          # saved .h5 / .pt / YOLO weights
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## Datasets & format

**Classification dataset (structure & counts)** — derived from the project brief:

* Train: bird (1414), drone (1248)
* Valid: bird (217), drone (225)
* Test: bird (121), drone (94)
  Images: RGB .jpg, recommended resize to 224×224 for typical transfer learning workflows. 

**Object detection dataset (YOLOv8 format)**:

* Total images: 3319 (train: 2662, val: 442, test: 215)
* Each annotation `.txt` line: `<class_id> <x_center> <y_center> <width> <height>` (normalized coordinates). 

---

## Quickstart — Installation

1. Clone the repo

```bash
git clone https://github.com/mangal-singh001/Aerial-Object-Classification-Detection.git
cd Aerial-Object-Classification-Detection
```

2. Create a Python environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

**Suggested `requirements.txt` basics**

```
numpy
pandas
matplotlib
scikit-learn
tensorflow>=2.10      # or torch if you use PyTorch
opencv-python
Pillow
streamlit
ultralytics           # for YOLOv8 (optional)
tqdm
seaborn
```

---

## How to run (examples)

### 1) Explore dataset (notebook)

Open `notebooks/01_explore_dataset.ipynb` in Jupyter / VSCode and run the cells to visualize class distribution and sample images.

### 2) Train a classification model (example)

Using a training script:

```bash
python src/train.py \
  --data_dir data/classification_dataset \
  --model resnet50 \
  --img_size 224 \
  --batch_size 32 \
  --epochs 30 \
  --save_dir models/
```

(See `src/train.py` for CLI options; notebook `03_train_classification.ipynb` contains step-by-step code.)

### 3) Evaluate

```bash
python src/evaluate.py --model_path models/best_model.h5 --test_dir data/classification_dataset/test
```

Outputs: confusion matrix, classification report (precision, recall, f1), and accuracy/loss curves saved to `models/` or `reports/`.

---

## Model training & evaluation — recommended approach

1. **Baseline**: Implement a small custom CNN (Conv → Pool → BN → Dropout → Dense) to get a baseline.
2. **Transfer learning**: Fine-tune a pretrained backbone (ResNet50 / EfficientNetB0 / MobileNetV2). Freeze base layers initially, then unfreeze and fine-tune.
3. **Callbacks**: Use `EarlyStopping`, `ModelCheckpoint` (save best val loss/accuracy), and learning-rate scheduler.
4. **Metrics**: Report accuracy, precision, recall, F1-score and the confusion matrix. Use class-weighting or augmentation if imbalance causes bias.
5. **Visualization**: Plot training/validation loss & accuracy, and show sample predictions with probabilities.

---

## YOLOv8 (optional) — object detection

If you want bounding-box detection (useful for real-time localization):

1. Install ultralytics (YOLOv8):

```bash
pip install ultralytics
```

2. Prepare `data.yaml` for YOLOv8:

```yaml
train: /path/to/object_detection_Dataset/images/train
val: /path/to/object_detection_Dataset/images/val
test: /path/to/object_detection_Dataset/images/test
names: ['bird', 'drone']
```

3. Train:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')   # choose a YOLOv8 variant
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)
```

4. Inference:

```python
results = model.predict(source='test_images/', conf=0.25, save=True)
```

---

## Streamlit deployment (classification + optional detection)

A simple UI to upload an image and show prediction:

1. Launch locally

```bash
cd streamlit_app
streamlit run app.py
```

2. `app.py` should:

* Load the best classification model once (cached).
* Accept uploaded image(s).
* Preprocess and show predictions with confidence.
* Optionally run YOLOv8 detection and display bounding boxes.

**Notes:** Use `@st.cache_resource` for model loading to speed the app.

---

## Results & deliverables

Deliverables to include in the repo:

* Trained classification model(s) (e.g., `models/best_model.h5`)
* Trained YOLOv8 weights (optional)
* Notebooks: preprocessing, training, evaluation, detection
* Streamlit app and requirements
* A short video demo (8+ minutes) explaining the app, demo, and notebook walkthrough (as mentioned in the project plan). 

> Tip: In your README you can add a small performance table (replace values after training):
> | Model | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 |
> |-------|-----------:|--------:|---------:|----------:|-------:|----:|
> | Custom CNN | 0.92 | 0.86 | 0.85 | 0.86 | 0.83 | 0.84 |
> | ResNet50 FT | 0.95 | 0.91 | 0.90 | 0.91 | 0.89 | 0.90 |

*(Fill with your real numbers after experiments.)*

---

## Tips & troubleshooting

* If classes are imbalanced, use augmentation, class weights or oversampling.
* If overfitting: increase dropout, add regularization, freeze more of the pretrained backbone, or use stronger augmentation.
* For faster iterations use a smaller image size (e.g., 128×128) during debugging, then move to 224×224 or higher for final runs.
* For inference speed (real-time), consider lightweight backbones (MobileNet/EfficientNet-lite) or smaller YOLOv8 variants (yolov8n/yolov8s).

---

## Contributing

Contributions, bug reports, and suggestions are welcome:

1. Fork the repo
2. Create a branch: `feat/your-feature`
3. Add changes / notebooks / improvements
4. Open a PR with a clear description

Please include unit tests or reproducible notebooks where possible.

---

## License & contact

This project is provided under the **MIT License** (or choose a license you prefer).
For questions or help, open an issue in the repository or contact the maintainer (GitHub: `mangal-singh001`).

---

## Acknowledgements

Project brief and dataset breakdown referenced from the project document. 

---

If you want, I can:

* generate a polished `README.md` file ready to paste into the repo, or
* produce the `streamlit_app/app.py` starter file and `src/train.py` skeleton next (with detailed code and comments) — tell me which to create first and I’ll produce it here.
