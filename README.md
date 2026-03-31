# A Comparative Study of Deep Learning Architectures for Multi-Disease Prediction from Medical Imaging and Clinical Data

## Abstract

This work presents a unified research platform for evaluating and comparing state-of-the-art deep learning architectures — **ResNet-50**, **VGG-16**, and **MobileNet V3** — across four distinct medical image classification tasks: retinal disease detection from OCT scans, brain tumor classification from MRI, pneumonia detection from chest X-rays, and malaria parasite identification from thin blood smear images. In addition, the platform incorporates classical machine learning baselines (Support Vector Machines) for three tabular clinical prediction tasks: diabetes, heart disease, and Parkinson's disease. The system is accompanied by a Flask-based interactive comparison dashboard that facilitates visual inspection of per-model performance metrics, confusion matrices, and per-class analysis across all seven disease domains.

**Total diseases covered:** 7 | **Total models evaluated:** 15 | **Total image samples:** 124,939+

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Datasets](#3-datasets)
4. [Model Architectures](#4-model-architectures)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Installation and Reproduction](#6-installation-and-reproduction)
7. [Usage](#7-usage)
8. [Project Structure](#8-project-structure)
9. [Dataset Download Links](#9-dataset-download-links)
10. [Pre-trained Model Sources](#10-pre-trained-model-sources)
11. [References](#11-references)
12. [Author](#12-author)

---

## 1. Introduction

Early and accurate disease detection remains one of the most impactful applications of machine learning in healthcare. While numerous studies have demonstrated the efficacy of convolutional neural networks (CNNs) for individual medical imaging tasks, few provide a consolidated framework for **cross-architecture, cross-disease comparative analysis**. This project addresses that gap by implementing a standardised evaluation pipeline across multiple disease domains and CNN architectures, enabling direct performance comparison under consistent experimental conditions.

The platform serves two purposes:
- **Research utility:** Systematic comparison of transfer learning performance across diseases of varying complexity, class count, and imaging modality.
- **Clinical demonstration:** An interactive web interface for real-time inference and metric visualisation, suitable for educational and presentation contexts.

---

## 2. System Architecture

The system is implemented as a **Flask** web application with Jinja2 templating, Bootstrap 5 for responsive layout, and Chart.js for client-side metric visualisation.

```
                    +-------------------+
                    |   Flask Server    |
                    |   (server.py)     |
                    +--------+----------+
                             |
          +------------------+------------------+
          |                  |                  |
   +------+------+   +------+------+   +-------+------+
   |  Prediction  |   |  Comparison |   |  Methodology |
   |   Engine     |   |  Dashboard  |   |   & Refs     |
   +------+------+   +------+------+   +--------------+
          |                  |
   +------+------+   +------+------+
   | Model Loader |   |   Metrics   |
   | (pickle/keras)|  |   Loader    |
   +--------------+   | (JSON files)|
                      +--------------+
```

**Backend modules:**
- `utils/model_loader.py` — Loads and caches pickle (SVM) and Keras (CNN) models
- `utils/prediction.py` — Executes tabular SVM inference and multi-model image classification
- `utils/preprocessing.py` — Architecture-specific image preprocessing (MobileNet, ResNet, VGG)
- `utils/metrics_loader.py` — Parses pre-computed evaluation metrics from JSON

**Frontend:**
- `templates/` — Jinja2 HTML: landing page, per-disease views (prediction + comparison tabs), cross-disease dashboard, methodology page
- `static/js/charts.js` — Chart.js: grouped bar charts, radar plots, confusion matrix heatmaps
- `static/js/main.js` — Image drag-and-drop upload, AJAX inference, tabular form handling
- `static/css/style.css` — Custom responsive styling

---

## 3. Datasets

### 3.1 Image Classification Datasets

| Disease | Dataset | Samples | Classes | Modality | Source |
|---------|---------|--------:|---------|----------|--------|
| Retinal Disease | Kermany OCT | 84,495 | 4 (CNV, DME, DRUSEN, NORMAL) | OCT | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| Brain Tumor | Brain Tumor MRI | 7,023 | 4 (Glioma, Meningioma, No Tumor, Pituitary) | MRI | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Pneumonia | Chest X-Ray | 5,863 | 2 (Normal, Pneumonia) | X-Ray | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Malaria | NIH Cell Images | 27,558 | 2 (Parasitized, Uninfected) | Microscopy | [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) |

### 3.2 Tabular Clinical Datasets

| Disease | Dataset | Samples | Features | Source |
|---------|---------|--------:|---------:|--------|
| Diabetes | Pima Indians Diabetes Database | 768 | 8 | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Heart Disease | Cleveland Heart Disease | 303 | 13 | [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| Parkinson's Disease | Oxford Parkinson's Detection | 195 | 22 | [Kaggle](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) |

---

## 4. Model Architectures

### 4.1 Convolutional Neural Networks (Image Tasks)

All CNN models employ **transfer learning** from ImageNet-pretrained weights with a custom classification head (Global Average Pooling, Dense 128-ReLU, Dropout, Softmax/Sigmoid output).

| Architecture | Year | Parameters | ImageNet Top-5 | Key Contribution |
|-------------|------|------------|-----------------|------------------|
| **VGG-16** | 2014 | 138M | 92.7% | Demonstrated that network depth with small (3x3) convolutional filters significantly improves classification performance |
| **ResNet-50** | 2015 | 25.6M | 93.3% | Introduced residual skip connections, enabling training of substantially deeper networks without degradation |
| **MobileNet V3** | 2019 | 5.4M | 92.6% | Combined neural architecture search with squeeze-and-excitation blocks for mobile-optimised inference |

### 4.2 Classical Machine Learning (Tabular Tasks)

| Algorithm | Kernel | Preprocessing | Validation |
|-----------|--------|---------------|------------|
| **Support Vector Machine** | Linear | StandardScaler | 80/20 stratified split |

---

## 5. Evaluation Metrics

All models are evaluated using the following standard classification metrics:

- **Accuracy** — Overall proportion of correct predictions
- **Precision** — Positive predictive value; penalises false positives
- **Recall (Sensitivity)** — True positive rate; critical in clinical screening where missed diagnoses carry high cost
- **F1 Score** — Harmonic mean of precision and recall; balances both error types
- **AUC-ROC** — Area under the Receiver Operating Characteristic curve; threshold-independent discriminative ability
- **Confusion Matrix** — Per-class breakdown of true positives, false positives, true negatives, and false negatives

Metrics are pre-computed and stored as structured JSON files in `metrics/`, enabling the comparison dashboard to render without requiring model re-evaluation.

---

## 6. Installation and Reproduction

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
git clone https://github.com/Vasuaggarwal0509/Multiple-disease-prediction-app.git
cd Multiple-disease-prediction-app

pip install -r requirements.txt
```

### Download Datasets

Datasets must be obtained from Kaggle (see [Section 9](#9-dataset-download-links)). Place test images in the corresponding `test_data/<disease>/` directory for inference demonstrations.

### Download or Train Models

Pre-trained model files (`.keras`) should be placed in `models/<disease>/`. Sources for pre-trained models are listed in [Section 10](#10-pre-trained-model-sources). Alternatively, train models using the notebooks provided in `notebooks/`.

```bash
# Check model download status
python scripts/download_models.py --list
```

---

## 7. Usage

### Run the Flask Application (Primary)

```bash
python server.py
# Access at http://localhost:5000
```

### Run the Legacy Streamlit Application

```bash
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
```

### Application Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with disease overview and navigation |
| `/disease/<key>` | Per-disease prediction interface with model comparison tab |
| `/comparison` | Cross-disease model comparison dashboard |
| `/methodology` | Evaluation protocol, metric definitions, and full bibliography |

---

## 8. Project Structure

```
Disease-prediction-system/
├── server.py                          # Flask application entry point
├── app.py                             # Legacy Streamlit application
├── requirements.txt
├── config/
│   ├── diseases.json                  # Disease metadata, feature definitions, model registry
│   └── references.json                # Structured research paper citations
├── models/
│   ├── diabetes/svm_model.sav
│   ├── heart/svm_model.sav
│   ├── parkinsons/svm_model.sav
│   ├── eye_disease/{mobilenet_v3,resnet50,vgg16}.keras
│   ├── brain_tumor/{mobilenet_v3,resnet50,vgg16}.keras
│   ├── pneumonia/{mobilenet_v3,resnet50,vgg16}.keras
│   └── malaria/{mobilenet_v3,resnet50,vgg16}.keras
├── metrics/                           # Pre-computed evaluation metrics (JSON)
│   ├── eye_disease/
│   ├── brain_tumor/
│   ├── pneumonia/
│   └── malaria/
├── templates/                         # Jinja2 HTML templates
├── static/                            # CSS, JavaScript, images
├── utils/                             # Backend modules
├── notebooks/                         # Model training notebooks
├── scripts/                           # Download and setup utilities
├── Datasets/                          # Tabular CSV datasets
├── test_data/                         # Sample images for inference
├── Saved_models/                      # Original SVM pickle files
├── Machine_Learning_models/           # Original training notebooks
└── Human_Eye_Disease_Prediction/      # Original eye disease Streamlit app
```

---

## 9. Dataset Download Links

All datasets are publicly available and can be downloaded from the following sources:

### Image Datasets

| Dataset | Link | Size | Notes |
|---------|------|------|-------|
| **Kermany OCT Retinal Images** | [https://www.kaggle.com/datasets/paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) | ~5.8 GB | 84,495 OCT images across 4 classes. Used in Kermany et al. (2018), *Cell*. |
| **Brain Tumor MRI Dataset** | [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | ~150 MB | 7,023 MRI images, 4 tumour categories. |
| **Chest X-Ray Pneumonia** | [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | ~1.2 GB | 5,863 chest X-rays, binary classification. |
| **NIH Malaria Cell Images** | [https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) | ~350 MB | 27,558 cell images from the NIH National Library of Medicine. |

### Tabular Datasets

| Dataset | Link |
|---------|------|
| **Pima Indians Diabetes** | [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Cleveland Heart Disease** | [https://www.kaggle.com/datasets/ronitf/heart-disease-uci](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| **Oxford Parkinson's** | [https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) |

---

## 10. Pre-trained Model Sources

The following Kaggle notebooks provide pre-trained model weights or reproducible training code for each disease-architecture combination. Fork and run these notebooks to obtain `.keras` model files, then place them in the corresponding `models/<disease>/` directory.

### Eye Disease (OCT)

| Model | Kaggle Notebooks |
|-------|-----------------|
| MobileNet V3 | [OCT Retinal Disease MobileNetV3](https://www.kaggle.com/code/arnavjain1/oct-retinal-disease-mobilenetv3) , [Detect Retina Damage from OCT Images](https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images) |
| ResNet-50 | [OCT Retinal Disease Classification ResNet50](https://www.kaggle.com/code/bjoernjostein/oct-retinal-disease-classification-resnet50) , [Retinal OCT ResNet50 Transfer Learning](https://www.kaggle.com/code/alifrahman/retinal-oct-resnet50-transfer-learning) |
| VGG-16 | [Retinal Disease Classification VGG16](https://www.kaggle.com/code/arifmia/retinal-disease-classification-vgg16) , [OCT Classification VGG16](https://www.kaggle.com/code/shauryasingh21/oct-classification-vgg16) |

### Brain Tumor (MRI)

| Model | Kaggle Notebooks |
|-------|-----------------|
| MobileNet V3 | [Brain Tumor MRI Classification MobileNet](https://www.kaggle.com/code/jaykumar1607/brain-tumor-mri-classification-mobilenet) , [Brain Tumor Detection MobileNetV3](https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-mobilenetv3) |
| ResNet-50 | [Brain Tumor Detection ResNet50](https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-resnet50) , [Brain Tumor Classification ResNet50](https://www.kaggle.com/code/mohammedaltet/brain-tumor-classification-resnet50) |
| VGG-16 | [Brain Tumor Detection VGG16](https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-vgg16) , [Brain Tumor Classification VGG16](https://www.kaggle.com/code/daniilkondratiev/brain-tumor-classification-vgg16) |

### Pneumonia (Chest X-Ray)

| Model | Kaggle Notebooks |
|-------|-----------------|
| MobileNet V3 | [Pneumonia Detection MobileNet](https://www.kaggle.com/code/jnegrini/pneumonia-detection-mobilenet) , [TensorFlow Pneumonia Classification](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays) |
| ResNet-50 | [Pneumonia Detection Using ResNet50](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-resnet50) , [Chest X-Ray ResNet50](https://www.kaggle.com/code/theimageprocessingguy/chest-xray-resnet50) |
| VGG-16 | [Pneumonia X-Ray VGG16](https://www.kaggle.com/code/sanwal092/pneumonia-x-ray-vgg16) , [Pneumonia Detection VGG16](https://www.kaggle.com/code/rishabhmohan/pneumonia-detection-vgg16) |

### Malaria (Cell Images)

| Model | Kaggle Notebooks |
|-------|-----------------|
| MobileNet V3 | [Malaria Cell Detection MobileNet](https://www.kaggle.com/code/akshat0007/malaria-cell-detection-mobilenet) , [Malaria Cell Classification](https://www.kaggle.com/code/shobhit18th/malaria-cell-classification) |
| ResNet-50 | [Malaria Detection ResNet50](https://www.kaggle.com/code/fchollet/malaria-detection-resnet50) , [Malaria Cell Images ResNet50](https://www.kaggle.com/code/harshwalia/malaria-cell-images-resnet50) |
| VGG-16 | [Malaria Detection VGG16](https://www.kaggle.com/code/therealsampat/malaria-detection-vgg16) , [Malaria VGG16 Transfer Learning](https://www.kaggle.com/code/gabrielrego/malaria-vgg16-transfer-learning) |

---

## 11. References

### Dataset Papers

1. Kermany, D.S., Goldbaum, M., Cai, W., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122-1131. DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)

2. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care, 261-265.

3. Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). *International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease.* American Journal of Cardiology, 64(5), 304-310. DOI: [10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9)

4. Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A., Moroz, I.M. (2009). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.* BioMedical Engineering OnLine, 6(1). DOI: [10.1186/1475-925X-6-23](https://doi.org/10.1186/1475-925X-6-23)

5. Rajaraman, S., Antani, S.K., Poostchi, M., et al. (2018). *Pre-trained Convolutional Neural Networks as Feature Extractors toward Improved Malaria Parasite Detection in Thin Blood Smear Images.* PeerJ, 6. DOI: [10.7717/peerj.4568](https://doi.org/10.7717/peerj.4568)

6. Nickparvar, M. (2021). *Brain Tumor MRI Dataset.* Kaggle.

### Model Architecture Papers

7. He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)

8. Simonyan, K., Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* International Conference on Learning Representations (ICLR). DOI: [10.48550/arXiv.1409.1556](https://doi.org/10.48550/arXiv.1409.1556)

9. Howard, A., Sandler, M., Chen, B., et al. (2019). *Searching for MobileNetV3.* IEEE/CVF International Conference on Computer Vision (ICCV). DOI: [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140)

### Related Work

10. Badza, M.M., Barjaktarovic, M.C. (2020). *Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network.* Applied Sciences, 10(6). DOI: [10.3390/app10061999](https://doi.org/10.3390/app10061999)

11. Deepak, S., Ameer, P.M. (2019). *Brain Tumor Classification Using Deep CNN Features via Transfer Learning.* Computers in Biology and Medicine, 111. DOI: [10.1016/j.compbiomed.2019.103345](https://doi.org/10.1016/j.compbiomed.2019.103345)

12. Rajpurkar, P., Irvin, J., Zhu, K., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv preprint. DOI: [10.48550/arXiv.1711.05225](https://doi.org/10.48550/arXiv.1711.05225)

13. Stephen, O., Sain, M., Maduh, U.J., Jeong, D.U. (2019). *An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare.* Journal of Healthcare Engineering. DOI: [10.1155/2019/4180949](https://doi.org/10.1155/2019/4180949)

14. Rajaraman, S., Jaeger, S., Antani, S.K. (2019). *Performance Evaluation of Deep Neural Ensembles toward Malaria Parasite Detection in Thin-Blood Smear Images.* PeerJ, 7. DOI: [10.7717/peerj.6977](https://doi.org/10.7717/peerj.6977)

---

## 12. Author

**Vasu Aggarwal**

For queries regarding this work, please open an issue on the repository or reach out via GitHub.

---

*This project is intended for educational and research purposes. It should not be used as a substitute for professional medical diagnosis.*
