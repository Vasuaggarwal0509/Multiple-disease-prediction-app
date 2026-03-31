"""
Download pre-trained models and test datasets for the Disease Prediction System.

This script provides URLs and instructions for downloading models from Kaggle.
Since Kaggle requires authentication, models must be downloaded manually or
via the Kaggle API.

Usage:
    # First, set up Kaggle API credentials:
    # pip install kaggle
    # Place kaggle.json in ~/.kaggle/

    python scripts/download_models.py --list          # List all required models
    python scripts/download_models.py --download-test # Download test datasets
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# MODEL SOURCES
# These are the Kaggle notebooks/datasets where you can find
# pre-trained models for each disease + architecture combo.
# ============================================================

MODEL_SOURCES = {
    "eye_disease": {
        "dataset": "https://www.kaggle.com/datasets/paultimothymooney/kermany2018",
        "models": {
            "mobilenet_v3": {
                "source": "Already trained (Human_Eye_Disease_Prediction/Trained_Model.keras)",
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/arnavjain1/oct-retinal-disease-mobilenetv3",
                    "https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images",
                ],
            },
            "resnet50": {
                "source": "Train from Kaggle notebook or download",
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/bjoernjostein/oct-retinal-disease-classification-resnet50",
                    "https://www.kaggle.com/code/alifrahman/retinal-oct-resnet50-transfer-learning",
                ],
            },
            "vgg16": {
                "source": "Train from Kaggle notebook or download",
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/arifmia/retinal-disease-classification-vgg16",
                    "https://www.kaggle.com/code/shauryasingh21/oct-classification-vgg16",
                ],
            },
        },
    },
    "brain_tumor": {
        "dataset": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "models": {
            "mobilenet_v3": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/jaykumar1607/brain-tumor-mri-classification-mobilenet",
                    "https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-mobilenetv3",
                ],
            },
            "resnet50": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-resnet50",
                    "https://www.kaggle.com/code/mohammedaltet/brain-tumor-classification-resnet50",
                ],
            },
            "vgg16": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/ahmedhamada0/brain-tumor-detection-vgg16",
                    "https://www.kaggle.com/code/daniilkondratiev/brain-tumor-classification-vgg16",
                ],
            },
        },
    },
    "pneumonia": {
        "dataset": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "models": {
            "mobilenet_v3": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/jnegrini/pneumonia-detection-mobilenet",
                    "https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays",
                ],
            },
            "resnet50": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/madz2000/pneumonia-detection-using-resnet50",
                    "https://www.kaggle.com/code/theimageprocessingguy/chest-xray-resnet50",
                ],
            },
            "vgg16": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/sanwal092/pneumonia-x-ray-vgg16",
                    "https://www.kaggle.com/code/rishabhmohan/pneumonia-detection-vgg16",
                ],
            },
        },
    },
    "malaria": {
        "dataset": "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
        "models": {
            "mobilenet_v3": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/akshat0007/malaria-cell-detection-mobilenet",
                    "https://www.kaggle.com/code/shobhit18th/malaria-cell-classification",
                ],
            },
            "resnet50": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/fchollet/malaria-detection-resnet50",
                    "https://www.kaggle.com/code/harshwalia/malaria-cell-images-resnet50",
                ],
            },
            "vgg16": {
                "kaggle_notebooks": [
                    "https://www.kaggle.com/code/therealsampat/malaria-detection-vgg16",
                    "https://www.kaggle.com/code/gabrielrego/malaria-vgg16-transfer-learning",
                ],
            },
        },
    },
}

# Expected model file paths
MODEL_FILES = {
    "eye_disease/mobilenet_v3": "models/eye_disease/mobilenet_v3.keras",
    "eye_disease/resnet50": "models/eye_disease/resnet50.keras",
    "eye_disease/vgg16": "models/eye_disease/vgg16.keras",
    "brain_tumor/mobilenet_v3": "models/brain_tumor/mobilenet_v3.keras",
    "brain_tumor/resnet50": "models/brain_tumor/resnet50.keras",
    "brain_tumor/vgg16": "models/brain_tumor/vgg16.keras",
    "pneumonia/mobilenet_v3": "models/pneumonia/mobilenet_v3.keras",
    "pneumonia/resnet50": "models/pneumonia/resnet50.keras",
    "pneumonia/vgg16": "models/pneumonia/vgg16.keras",
    "malaria/mobilenet_v3": "models/malaria/mobilenet_v3.keras",
    "malaria/resnet50": "models/malaria/resnet50.keras",
    "malaria/vgg16": "models/malaria/vgg16.keras",
}


def list_models():
    """List all required models and their download status."""
    print("=" * 70)
    print("DISEASE PREDICTION SYSTEM - MODEL STATUS")
    print("=" * 70)

    for key, path in MODEL_FILES.items():
        full_path = os.path.join(BASE_DIR, path)
        exists = os.path.exists(full_path)
        status = "FOUND" if exists else "MISSING"
        icon = "[+]" if exists else "[-]"
        print(f"  {icon} {key:40s} {status}")

    print("\n" + "=" * 70)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 70)

    for disease, info in MODEL_SOURCES.items():
        print(f"\n--- {disease.upper().replace('_', ' ')} ---")
        print(f"  Dataset: {info['dataset']}")
        for model, details in info["models"].items():
            print(f"\n  {model}:")
            if "source" in details:
                print(f"    Source: {details['source']}")
            print(f"    Kaggle notebooks to use:")
            for url in details["kaggle_notebooks"]:
                print(f"      - {url}")
            target = f"models/{disease}/{model}.keras"
            print(f"    Save as: {target}")

    print("\n" + "=" * 70)
    print("HOW TO DOWNLOAD:")
    print("  1. Open the Kaggle notebook links above")
    print("  2. Run the notebook (or fork and run)")
    print("  3. Download the trained model file from the notebook output")
    print("  4. Save it to the path shown above")
    print("  OR: Train your own using notebooks in notebooks/ directory")
    print("=" * 70)


def check_test_data():
    """Check if test data directories have images."""
    print("\nTEST DATA STATUS:")
    for disease in ["eye_disease", "brain_tumor", "pneumonia", "malaria"]:
        test_dir = os.path.join(BASE_DIR, "test_data", disease)
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if not f.startswith(".")]
            print(f"  {disease}: {len(files)} files")
        else:
            print(f"  {disease}: Directory empty (add test images here)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model download helper")
    parser.add_argument("--list", action="store_true", help="List all models and status")
    parser.add_argument("--check", action="store_true", help="Check test data status")
    args = parser.parse_args()

    if args.check:
        check_test_data()
    else:
        list_models()
        check_test_data()
