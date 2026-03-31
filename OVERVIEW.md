# Multi-Disease Prediction System - Quick Overview

A web app that predicts 7 diseases using Machine Learning. It compares 3 deep learning models (ResNet-50, VGG-16, MobileNet V3) on medical images and uses SVM for clinical data.

---

## What Does This Project Do?

You upload a medical image (X-ray, MRI, OCT scan, or cell image) and the app runs **3 different AI models** on it simultaneously, showing you which disease it detects and how confident each model is. It also has a **comparison dashboard** showing which model performs best.

For clinical data (diabetes, heart disease, Parkinson's), you enter patient values and get an instant prediction.

---

## Diseases Covered

### Image-Based (3 AI models each)

| Disease | Input | Classes | Dataset Size |
|---------|-------|---------|-------------|
| Eye Disease | OCT retinal scan | CNV, DME, DRUSEN, Normal | 84,495 images |
| Brain Tumor | MRI scan | Glioma, Meningioma, Pituitary, No Tumor | 7,023 images |
| Pneumonia | Chest X-ray | Normal, Pneumonia | 5,863 images |
| Malaria | Blood cell image | Parasitized, Uninfected | 27,558 images |

### Clinical Data (SVM model)

| Disease | Input | Accuracy |
|---------|-------|----------|
| Diabetes | 8 medical values (glucose, BMI, age, etc.) | 77% |
| Heart Disease | 13 cardiac values (cholesterol, BP, etc.) | 87% |
| Parkinson's | 22 voice measurements | 87% |

---

## How to Run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python server.py

# 3. Open browser
# http://localhost:5000
```

---

## Download Links

### Datasets (from Kaggle)

| Dataset | Download |
|---------|----------|
| Eye Disease (OCT) | https://www.kaggle.com/datasets/paultimothymooney/kermany2018 |
| Brain Tumor (MRI) | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| Pneumonia (X-Ray) | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Malaria (Cell) | https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria |
| Diabetes | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Heart Disease | https://www.kaggle.com/datasets/ronitf/heart-disease-uci |
| Parkinson's | https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set |

### Pre-trained Models (Kaggle Notebooks - fork and run to get .keras files)

**Eye Disease**
- MobileNet V3: https://www.kaggle.com/code/arnavjain1/oct-retinal-disease-mobilenetv3
- ResNet-50: https://www.kaggle.com/code/bjoernjostein/oct-retinal-disease-classification-resnet50
- VGG-16: https://www.kaggle.com/code/arifmia/retinal-disease-classification-vgg16

**Brain Tumor**
- MobileNet V3: https://www.kaggle.com/code/jaykumar1607/brain-tumor-mri-classification-mobilenet
- ResNet-50: https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-resnet50
- VGG-16: https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-vgg16

**Pneumonia**
- MobileNet V3: https://www.kaggle.com/code/jnegrini/pneumonia-detection-mobilenet
- ResNet-50: https://www.kaggle.com/code/madz2000/pneumonia-detection-using-resnet50
- VGG-16: https://www.kaggle.com/code/sanwal092/pneumonia-x-ray-vgg16

**Malaria**
- MobileNet V3: https://www.kaggle.com/code/akshat0007/malaria-cell-detection-mobilenet
- ResNet-50: https://www.kaggle.com/code/fchollet/malaria-detection-resnet50
- VGG-16: https://www.kaggle.com/code/therealsampat/malaria-detection-vgg16

### Research Papers

| Paper | Link |
|-------|------|
| ResNet (He et al., 2016) | https://doi.org/10.1109/CVPR.2016.90 |
| VGG (Simonyan & Zisserman, 2015) | https://doi.org/10.48550/arXiv.1409.1556 |
| MobileNet V3 (Howard et al., 2019) | https://doi.org/10.1109/ICCV.2019.00140 |
| OCT Dataset (Kermany et al., 2018) | https://doi.org/10.1016/j.cell.2018.02.010 |
| CheXNet - Pneumonia (Rajpurkar et al., 2017) | https://doi.org/10.48550/arXiv.1711.05225 |
| Malaria Detection (Rajaraman et al., 2018) | https://doi.org/10.7717/peerj.4568 |
| Brain Tumor CNN (Badza & Barjaktarovic, 2020) | https://doi.org/10.3390/app10061999 |

---

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript, Bootstrap 5, Chart.js
- **ML Models:** TensorFlow/Keras (CNNs), scikit-learn (SVM)
- **CNN Architectures:** ResNet-50, VGG-16, MobileNet V3

---

## Where to Put Downloaded Files

```
models/
  eye_disease/mobilenet_v3.keras    <-- put .keras files here
  eye_disease/resnet50.keras
  eye_disease/vgg16.keras
  brain_tumor/mobilenet_v3.keras
  brain_tumor/resnet50.keras
  brain_tumor/vgg16.keras
  pneumonia/mobilenet_v3.keras
  pneumonia/resnet50.keras
  pneumonia/vgg16.keras
  malaria/mobilenet_v3.keras
  malaria/resnet50.keras
  malaria/vgg16.keras

test_data/
  eye_disease/                      <-- put sample test images here
  brain_tumor/
  pneumonia/
  malaria/
```

---

## Author

**Vasu Aggarwal**
