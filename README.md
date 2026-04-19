# Multi-Disease Prediction System

A Flask-based web application providing AI-powered predictions for six medical conditions. Each disease is handled by a dedicated model trained on a specialized dataset — three Support Vector Machines for clinical tabular data and three Convolutional Neural Networks for medical imaging.

**6 diseases · 6 trained models · 117K+ training images · 2 data modalities**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Diseases Covered](#2-diseases-covered)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Datasets](#5-datasets)
6. [Model Architectures](#6-model-architectures)
7. [Training the Models](#7-training-the-models)
8. [Evaluation and Visualisation](#8-evaluation-and-visualisation)
9. [Measured Results](#9-measured-results)
10. [References](#10-references)
11. [License](#11-license)

---

## 1. Overview

This project provides:

- **A unified web interface** for uploading medical images or entering clinical values
- **Six pre-trained models** ready for inference out of the box
- **Jupyter notebooks** that document how each model was trained, for reproducibility
- **A reproducible evaluation harness** (`scripts/evaluate.py`) that measures accuracy, precision, recall, F1, and AUC-ROC on a labelled test set
- **Auto-generated visualisations** of the results for inclusion in reports

Unlike benchmark-comparison platforms, this system ships one purpose-trained model per disease. Each prediction returns a class label with per-class probability distribution.

---

## 2. Diseases Covered

### Image-Based (CNN Inference)

| Disease | Dataset | Classes | Model | Input Size |
|---------|---------|---------|-------|-----------:|
| Retinal Disease (OCT) | Kermany OCT (84,495 images) | CNV, DME, DRUSEN, NORMAL | MobileNet V3 (fine-tuned) | 224 × 224 |
| Pneumonia (Chest X-Ray) | Kermany Chest X-Ray (5,863 images) | Normal, Pneumonia | Custom CNN (5 Conv blocks) | 300 × 300 |
| Malaria (Cell Images) | NIH Malaria Cell Images (27,558 images) | Parasitized, Uninfected | Custom CNN (3 Conv blocks) | 130 × 130 |

### Tabular Clinical Data (SVM)

| Disease | Dataset | Samples | Features | Classes |
|---------|---------|--------:|---------:|---------|
| Diabetes | Pima Indians Diabetes Database | 768 | 8 | Not Diabetic, Diabetic |
| Heart Disease | Cleveland Heart Disease | 303 | 13 | No Heart Disease, Heart Disease |
| Parkinson's Disease | Oxford Parkinson's Voice Dataset | 195 | 22 | Healthy, Parkinson's |

---

## 3. Quick Start

### Prerequisites
- Python 3.11
- `uv` or `pip` for dependency management

### Installation

```bash
git clone <repository-url>
cd Disease-prediction-system

# Install dependencies
pip install -r requirements.txt
# or with uv
uv pip install -r requirements.txt
```

### Run the App

```bash
python server.py
```

Open `http://localhost:5000` in your browser.

### Usage

- **Image diseases:** Navigate to Eye Disease, Pneumonia, or Malaria. Drag-and-drop an image and click *Predict*.
- **Tabular diseases:** Navigate to Diabetes, Heart Disease, or Parkinson's. Fill in clinical values and submit.

Sample test images are provided in `test_data/<disease>/<class>/`.

---

## 4. Project Structure

```
Disease-prediction-system/
├── server.py                      # Flask application entry point
├── requirements.txt
├── README.md
├── OVERVIEW.md
├── config/
│   ├── diseases.json              # Disease metadata + model file paths
│   └── references.json            # Research paper citations
├── ml_models/                     # All trained model files
│   ├── diabetes/svm_model.sav
│   ├── heart/svm_model.sav
│   ├── parkinsons/svm_model.sav
│   ├── eye_disease/
│   │   ├── trained_model.keras
│   │   └── trained_model.h5       # fallback for older Keras
│   ├── pneumonia/trained_model.h5
│   └── malaria/trained_model.h5
├── utils/
│   ├── prediction.py              # SVM inference for tabular data
│   └── metrics_loader.py          # Config loaders
├── templates/                     # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── disease.html
│   └── methodology.html
├── static/
│   ├── css/style.css
│   └── js/main.js                 # Image upload + prediction logic
├── scripts/
│   ├── evaluate.py                # Reproducible evaluation harness
│   ├── visualize_results.py       # Plot generator from evaluation JSON
│   └── generate_report.py         # Build the formal report.docx
├── notebooks/                     # Training notebooks (reproducibility)
│   ├── Diabetes_prediction_system.ipynb
│   ├── Heart_disease_prediction_syestem.ipynb
│   ├── Parkinsons_prediction.ipynb
│   ├── eye_disease_training.ipynb
│   ├── pneumonia_training.ipynb
│   └── malaria_training.ipynb
├── Datasets/                      # CSVs for tabular SVM training
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── test_data/                     # Labelled evaluation images
│   ├── eye_disease/{CNV,DME,DRUSEN,NORMAL}/
│   ├── pneumonia/{Normal,Pneumonia}/
│   ├── malaria/{Parasitized,Uninfected}/
│   ├── ground_truth.csv           # auto-generated from folder layout
│   └── plots/                     # auto-generated visualisations
└── research_papers/               # Reference papers (gitignored)
```

---

## 5. Datasets

All datasets are publicly available on Kaggle. Download directly via the browser or the Kaggle API.

### Image Datasets

| Dataset | Size | Source |
|---------|-----:|--------|
| Kermany OCT Retinal Images | ~5.8 GB | [kaggle.com/datasets/paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| Kermany Chest X-Ray (Pneumonia) | ~1.2 GB | [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| NIH Malaria Cell Images | ~350 MB | [kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |

### Tabular Datasets (already included in `Datasets/`)

| Dataset | Source |
|---------|--------|
| Pima Indians Diabetes | [kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Cleveland Heart Disease | [kaggle.com/datasets/ronitf/heart-disease-uci](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| Oxford Parkinson's | [kaggle.com/datasets/vikasukani/parkinsons-disease-data-set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) |

---

## 6. Model Architectures

### Image Models

**Eye Disease (OCT) — MobileNet V3 (fine-tuned)**
- Base: `tf.keras.applications.MobileNetV3Large` with ImageNet weights, `include_preprocessing=True`
- Head: `Dense(4, softmax)` for 4-class classification
- Training: 15 epochs, Adam (lr=1e-4), categorical cross-entropy
- Preprocessing: MobileNet V3 preprocess_input ([-1, 1] scaling, applied internally)

**Pneumonia (Chest X-Ray) — Custom CNN**
- Five `Conv2D → MaxPooling2D` blocks (filters: 16 → 32 → 64 → 128 → 128)
- `Flatten → Dense(256) → Dense(512) → Dense(1, sigmoid)`
- Binary cross-entropy, Adam optimizer
- Preprocessing: rescale by 1/255

**Malaria (Cell Images) — Custom CNN**
- Three `Conv2D → MaxPooling2D` blocks (filters: 32 → 64 → 64)
- `Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid)`
- Binary cross-entropy, Adam optimizer
- Preprocessing: rescale by 1/255

### Tabular Models

All three tabular diseases use **Support Vector Machine with linear kernel** (`sklearn.svm.SVC`), trained on the raw CSV features with StandardScaler preprocessing during training and an 80/20 train-test split.

---

## 7. Training the Models

Each disease has a Jupyter notebook in `notebooks/` that reproduces the deployed model. The image-training notebooks are designed for Google Colab or Kaggle (GPU recommended).

### Steps

1. Download the corresponding dataset from the Kaggle link in [Section 5](#5-datasets).
2. Open the training notebook (e.g., `notebooks/eye_disease_training.ipynb`) in Colab or Jupyter.
3. Adjust the dataset paths in the Configuration cell.
4. Run all cells. Training takes 10–60 minutes depending on dataset size and GPU.
5. Copy the saved model file(s) into the corresponding `ml_models/<disease>/` directory.

The Flask app automatically loads models from these paths on first prediction.

---

## 8. Evaluation and Visualisation

Two reproducible scripts drive the evaluation pipeline.

### Run the evaluation

Drop ~20 labelled images per class into `test_data/<disease>/<class>/`, then:

```bash
python scripts/evaluate.py
```

This single command:

1. Scans every image file in `test_data/<disease>/<class>/` and writes `test_data/ground_truth.csv`.
2. Loads each of the three trained image models.
3. Runs inference on every image, applying the same preprocessing as the live server.
4. Computes accuracy, precision, recall, F1 score, AUC-ROC, per-class metrics, and confusion matrices.
5. Saves everything to `test_data/evaluation_results.json`.

### Generate plots

```bash
python scripts/visualize_results.py
```

Reads `evaluation_results.json` and emits eleven PNG plots into `test_data/plots/` — one confusion-matrix heatmap, one all-metrics bar chart, and one per-class breakdown for each disease, plus two cross-disease summaries.

### Generate the report

```bash
python scripts/generate_report.py
```

Builds the formal `report.docx` file. Uses the existing title page, abstract, certificate, and acknowledgement from `finalthesis.docx`, then rebuilds the seven chapters with current architecture, real metrics, and the per-class plots.

---

## 9. Measured Results

Held-out evaluation on labelled test images (20 per class):

| Disease | Model | Accuracy | Precision | Recall | F1 | Test Set |
|---------|-------|:--------:|:---------:|:------:|:--:|:--------:|
| Retinal Disease (OCT) | MobileNet V3 | 90.00% | 90.38% | 90.00% | 89.85% | 80 images |
| Pneumonia (Chest X-Ray) | Custom CNN | 97.50% | 95.24% | 100.00% | 97.56% | 40 images |
| Malaria (Cell Images) | Custom CNN | 97.50% | 95.24% | 100.00% | 97.56% | 40 images |

Tabular SVM evaluations (20% held-out split from training):

| Disease | Accuracy | Test Set |
|---------|:--------:|:--------:|
| Diabetes | 77.27% | ≈ 154 records |
| Heart Disease | 86.89% | ≈ 61 records |
| Parkinson's Disease | 87.18% | 39 records |

---

## 10. References

### Dataset Papers
- Kermany, D.S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122–1131. DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
- Smith, J.W., et al. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care.
- Detrano, R., et al. (1989). *International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease.* American Journal of Cardiology, 64(5), 304–310.
- Little, M.A., et al. (2009). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.* BioMedical Engineering OnLine, 6(1).
- Rajaraman, S., et al. (2018). *Pre-trained Convolutional Neural Networks as Feature Extractors toward Improved Malaria Parasite Detection in Thin Blood Smear Images.* PeerJ, 6.

### Model Architecture Papers
- Howard, A., et al. (2019). *Searching for MobileNetV3.* IEEE/CVF International Conference on Computer Vision. DOI: [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140)

### Related Work
- Rajpurkar, P., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.
- Stephen, O., et al. (2019). *An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare.* Journal of Healthcare Engineering.

---

## 11. License

This project is intended for educational and research purposes. It is **not a substitute for professional medical diagnosis**. Always consult a qualified healthcare provider for clinical decisions.

---

## Author

**Vasu Aggarwal**

Submit issues or pull requests via the repository.
