"""
Create transfer learning model architectures for all diseases.
Uses ImageNet-pretrained weights as the base.
These models have correct architecture but are NOT fine-tuned on disease data.
Replace with properly trained models for real predictions.
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Disease configs: name -> num_classes
DISEASES = {
    "eye_disease": 4,    # CNV, DME, DRUSEN, NORMAL
    "brain_tumor": 4,    # Glioma, Meningioma, No Tumor, Pituitary
    "pneumonia": 2,      # Normal, Pneumonia
    "malaria": 2,        # Parasitized, Uninfected
}

INPUT_SHAPE = (224, 224, 3)


def build_model(base_model_fn, num_classes, preprocess_fn=None):
    """Build a transfer learning model with classification head."""
    base = base_model_fn(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False  # Freeze base

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = inputs
    if preprocess_fn:
        x = preprocess_fn(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    if num_classes == 2:
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def create_and_save(disease, model_name, base_fn, preprocess_fn, num_classes):
    """Create model and save to disk."""
    out_path = os.path.join(MODELS_DIR, disease, f"{model_name}.keras")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  [SKIP] {disease}/{model_name}.keras already exists ({size_mb:.1f}MB)")
        return

    print(f"  Building {model_name} for {disease} ({num_classes} classes)...")
    model = build_model(base_fn, num_classes, preprocess_fn)
    model.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  [DONE] Saved {out_path} ({size_mb:.1f}MB)")


def main():
    from tensorflow.keras.applications import (
        MobileNetV3Large,
        ResNet50,
        VGG16,
    )
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as resnet_vgg_preprocess

    model_configs = {
        "mobilenet_v3": (MobileNetV3Large, mobilenet_preprocess),
        "resnet50": (ResNet50, resnet_preprocess),
        "vgg16": (VGG16, resnet_vgg_preprocess),
    }

    for disease, num_classes in DISEASES.items():
        print(f"\n=== {disease.upper()} ({num_classes} classes) ===")
        for model_name, (base_fn, preprocess_fn) in model_configs.items():
            # Skip eye_disease/mobilenet_v3 since we already have the trained one
            if disease == "eye_disease" and model_name == "mobilenet_v3":
                print(f"  [SKIP] {disease}/{model_name} - using existing trained model")
                continue
            create_and_save(disease, model_name, base_fn, preprocess_fn, num_classes)

    print("\n=== ALL MODELS CREATED ===")
    print("Note: These use ImageNet pretrained weights (not fine-tuned on disease data).")
    print("For real predictions, replace with models trained on the actual datasets.")
    print("The comparison viewer uses pre-computed metrics from metrics/ directory.")


if __name__ == "__main__":
    main()
