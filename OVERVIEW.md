# Multi-Disease Prediction System — Quick Overview

A web app that predicts 6 diseases using AI. Three use clinical numeric data (SVM models), three use medical images (CNN models).

---

## What It Does

- **Upload a medical image** (OCT scan, chest X-ray, or blood cell image) → get a diagnosis
- **Enter patient values** (glucose, cholesterol, voice metrics) → get a risk assessment
- Clean Flask web UI with drag-and-drop uploads
- Reproducible evaluation harness + auto-generated plots for every model

Each disease has **one dedicated trained model**. No comparisons, no model picker — just predict.

---

## Diseases Covered

### Image-Based (3 CNN Models)

| Disease | Input | Classes | Model |
|---------|-------|---------|-------|
| Eye Disease | OCT retinal scan | CNV, DME, DRUSEN, NORMAL | MobileNet V3 (fine-tuned) |
| Pneumonia | Chest X-ray | Normal, Pneumonia | Custom CNN |
| Malaria | Blood cell image | Parasitized, Uninfected | Custom CNN |

### Clinical Data (3 SVM Models)

| Disease | Input | Classes |
|---------|-------|---------|
| Diabetes | 8 clinical values (glucose, BMI, age, etc.) | Not Diabetic / Diabetic |
| Heart Disease | 13 cardiac values (cholesterol, BP, etc.) | No Heart Disease / Heart Disease |
| Parkinson's Disease | 22 voice measurements | Healthy / Parkinson's |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or
uv pip install -r requirements.txt

# 2. Start the server
python server.py

# 3. Open browser
# http://localhost:5000
```

---

## Measured Results (on held-out test images)

| Disease | Model | Accuracy |
|---------|-------|:--------:|
| Retinal Disease (OCT) | MobileNet V3 | 90.00% |
| Pneumonia (Chest X-Ray) | Custom CNN | 97.50% |
| Malaria (Cell Images) | Custom CNN | 97.50% |
| Diabetes | SVM (linear) | 77.27% |
| Heart Disease | SVM (linear) | 86.89% |
| Parkinson's | SVM (linear) | 87.18% |

---

## Project Layout

```
ml_models/          <-- All trained model files (download or train yourself)
  diabetes/svm_model.sav
  heart/svm_model.sav
  parkinsons/svm_model.sav
  eye_disease/trained_model.keras (+ .h5 fallback)
  pneumonia/trained_model.h5
  malaria/trained_model.h5

notebooks/          <-- Training code (6 Jupyter notebooks)
scripts/            <-- evaluate.py, visualize_results.py, generate_report.py
test_data/          <-- Labelled test images (per disease/class) + ground_truth.csv + plots/
Datasets/           <-- CSVs used to train the SVM models
templates/          <-- HTML (Flask Jinja2)
static/             <-- CSS + JS
utils/              <-- Python backend modules
server.py           <-- Flask entry point
config/diseases.json <-- All disease metadata + model paths
```

---

## Dataset Download Links

| Dataset | Kaggle |
|---------|--------|
| Eye Disease (OCT, 84K images) | https://www.kaggle.com/datasets/paultimothymooney/kermany2018 |
| Pneumonia (Chest X-Ray) | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Malaria (Cell Images) | https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria |
| Diabetes | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Heart Disease | https://www.kaggle.com/datasets/ronitf/heart-disease-uci |
| Parkinson's | https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set |

---

## Evaluate the Models

Drop ~20 labelled images per class into `test_data/<disease>/<class>/`, then run:

```bash
# Scan test_data/ → compute metrics → save JSON
python scripts/evaluate.py

# Generate confusion matrices + per-class bar charts
python scripts/visualize_results.py

# Rebuild the formal report.docx with all plots embedded
python scripts/generate_report.py
```

Outputs:
- `test_data/ground_truth.csv` — auto-labelled from folder structure
- `test_data/evaluation_results.json` — accuracy, precision, recall, F1, AUC-ROC per disease
- `test_data/plots/*.png` — 11 visualisations (confusion matrices, per-class metrics, summaries)

---

## Training a Model Yourself

1. Open the notebook for your disease (e.g. `notebooks/eye_disease_training.ipynb`)
2. Upload to Google Colab or Kaggle (GPU recommended for image models)
3. Download the dataset (link above)
4. Adjust the paths in the Configuration cell
5. Run all cells
6. Download the saved model file
7. Drop it into the matching folder under `ml_models/<disease>/`

That's it — next time you start the server, the new model loads automatically.

---

## Tech Stack

- **Backend:** Python 3.11, Flask
- **ML:** TensorFlow/Keras (CNN models), scikit-learn (SVM)
- **Frontend:** HTML/CSS/JavaScript with Bootstrap 5
- **Data:** Publicly available Kaggle datasets

---

## Important Notes

- **TensorFlow version:** Pin to 2.14–2.15. Keras 3.x (TF 2.16+) cannot load the eye disease model due to legacy layer-name conventions.
- **Models are downloadable / reproducible:** If any model is missing, the server logs a clear warning. Train using the notebooks or download from a teammate.
- **Not for clinical use:** Educational and research purposes only.

---

## Author

**Vasu Aggarwal**
