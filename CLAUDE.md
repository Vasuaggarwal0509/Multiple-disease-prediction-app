# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research-grade multi-disease prediction platform with two frontends:
1. **Flask app** (`server.py`) — Unified web app with HTML/CSS/JS UI. Covers all 7 diseases: Diabetes, Heart, Parkinson's (tabular/SVM), plus Eye Disease, Brain Tumor, Pneumonia, Malaria (image/CNN). Includes model comparison viewer, metrics dashboards, and research references.
2. **Legacy Streamlit app** (`app.py`) — Original single-file Streamlit app for the 3 tabular diseases. Still functional.

## Common Commands

```bash
# Run the Flask app (main)
python server.py
# Runs on http://localhost:5000

# Run the legacy Streamlit app
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false

# Install dependencies
pip install -r requirements.txt

# Check model download status
python scripts/download_models.py --list
```

## Architecture

### Flask App (server.py)
- **server.py** — Flask routes: index, disease pages, prediction API endpoints, comparison dashboard, methodology page
- **templates/** — Jinja2 HTML templates: `base.html` (layout + Bootstrap 5 navbar), `index.html` (landing page with disease cards), `disease.html` (per-disease prediction + comparison tabs), `comparison.html` (cross-disease dashboard), `methodology.html` (research references)
- **static/** — `css/style.css` (custom styles), `js/main.js` (image upload, tabular forms, AJAX predictions), `js/charts.js` (Chart.js visualizations for metrics, confusion matrices, radar charts)
- **utils/** — `model_loader.py` (loads pickle/keras models with caching), `prediction.py` (tabular and multi-model image inference), `metrics_loader.py` (loads pre-computed JSON metrics), `preprocessing.py` (image preprocessing per CNN architecture)

### Data & Models
- **config/** — `diseases.json` (disease metadata, feature lists, model registry), `references.json` (research paper citations)
- **models/** — Organized by disease: `models/{disease}/{model_name}.keras` or `.sav`. Image diseases have 3 models each (MobileNet V3, ResNet-50, VGG-16).
- **metrics/** — Pre-computed JSON files per model: `metrics/{disease}/{model_key}_metrics.json`. Contains accuracy, precision, recall, F1, AUC-ROC, confusion matrices, per-class metrics.
- **Saved_models/** — Original SVM pickle files (legacy, also copied into `models/`)
- **Datasets/** — Training CSVs for tabular diseases
- **test_data/** — Small test image sets for live inference demos

### Image Diseases (CNN comparison)
- Eye Disease (OCT): 84K images, 4 classes (CNV, DME, DRUSEN, NORMAL)
- Brain Tumor (MRI): 7K images, 4 classes (Glioma, Meningioma, No Tumor, Pituitary)
- Pneumonia (Chest X-Ray): 5.8K images, 2 classes (Normal, Pneumonia)
- Malaria (Cell Images): 27K images, 2 classes (Parasitized, Uninfected)

### Tabular Diseases (SVM)
- Diabetes: 768 samples, 8 features
- Heart Disease: 303 samples, 13 features
- Parkinson's: 195 samples, 22 features

## Key Details

- Python 3.11 (defined in `.devcontainer/devcontainer.json`)
- Flask app loads disease config from `config/diseases.json` at startup
- Image preprocessing differs per model architecture (see `utils/preprocessing.py`): MobileNet uses [-1,1] scaling, ResNet/VGG use ImageNet mean subtraction (caffe mode)
- Models are lazy-loaded and cached in `utils/model_loader.py`
- Pre-computed metrics drive the comparison viewer — no live training needed
- Chart.js renders all visualizations client-side from metrics JSON served via `/api/metrics/<disease_key>`
- Tabular predictions hit `/predict/tabular` (JSON API), image predictions hit `/predict/image` (multipart form)
- No test suite; validation done via Jupyter notebooks and pre-computed metrics
