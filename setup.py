"""
Setup script for the Multi-Disease Prediction System.

Downloads and builds all CNN model files required by the application.
Models use ImageNet pre-trained weights with disease-specific classification heads.

Usage:
    python setup.py

This will create 12 .keras model files (~650MB total) in the models/ directory.
Requires: tensorflow, numpy
"""

import os
import sys
import time
import logging

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Disease definitions ──
DISEASES = {
    "eye_disease":  {"name": "Eye Disease (OCT)",       "classes": 4},
    "brain_tumor":  {"name": "Brain Tumor (MRI)",       "classes": 4},
    "pneumonia":    {"name": "Pneumonia (X-Ray)",       "classes": 2},
    "malaria":      {"name": "Malaria (Cell Images)",   "classes": 2},
}

ARCHITECTURES = ["mobilenet_v3", "resnet50", "vgg16"]
INPUT_SHAPE = (224, 224, 3)


def check_dependencies():
    """Verify required packages are installed."""
    log.info("Checking dependencies...")
    try:
        import tensorflow as tf
        log.info(f"  TensorFlow {tf.__version__} found")
    except ImportError:
        log.error("TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    try:
        import numpy as np
        log.info(f"  NumPy {np.__version__} found")
    except ImportError:
        log.error("NumPy is not installed. Run: pip install numpy")
        sys.exit(1)


def get_existing_models():
    """Return set of model keys that already exist on disk."""
    existing = set()
    for disease in DISEASES:
        for arch in ARCHITECTURES:
            path = os.path.join(MODELS_DIR, disease, f"{arch}.keras")
            if os.path.exists(path):
                existing.add(f"{disease}/{arch}")
    return existing


def build_model(base_model_fn, num_classes):
    """Build a transfer learning model with classification head."""
    from tensorflow import keras
    from tensorflow.keras import layers

    base = base_model_fn(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    activation = "sigmoid" if num_classes == 2 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation)(x)

    return keras.Model(inputs, outputs)


def create_model(disease_key, arch_key, num_classes):
    """Create and save a single model. Returns file size in MB."""
    from tensorflow.keras.applications import MobileNetV3Large, ResNet50, VGG16

    arch_map = {
        "mobilenet_v3": MobileNetV3Large,
        "resnet50": ResNet50,
        "vgg16": VGG16,
    }

    out_dir = os.path.join(MODELS_DIR, disease_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{arch_key}.keras")

    model = build_model(arch_map[arch_key], num_classes)
    model.save(out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    return size_mb


def main():
    print()
    print("=" * 60)
    print("  Multi-Disease Prediction System — Model Setup")
    print("=" * 60)
    print()

    check_dependencies()

    # Suppress TF warnings during model creation
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    existing = get_existing_models()
    total_models = len(DISEASES) * len(ARCHITECTURES)
    existing_count = len(existing)

    if existing_count == total_models:
        log.info(f"All {total_models} models already exist. Nothing to do.")
        print_summary()
        return

    log.info(f"Models found: {existing_count}/{total_models}")
    log.info(f"Models to create: {total_models - existing_count}")
    print()

    created = 0
    skipped = 0
    total_size = 0
    start_time = time.time()

    for disease_key, disease_info in DISEASES.items():
        log.info(f"── {disease_info['name']} ({disease_info['classes']} classes) ──")

        for arch_key in ARCHITECTURES:
            model_id = f"{disease_key}/{arch_key}"

            if model_id in existing:
                path = os.path.join(MODELS_DIR, disease_key, f"{arch_key}.keras")
                size = os.path.getsize(path) / (1024 * 1024)
                log.info(f"  [SKIP] {arch_key:15s}  already exists ({size:.1f} MB)")
                skipped += 1
                total_size += size
                continue

            log.info(f"  [BUILD] {arch_key:15s}  downloading weights & building...")
            t0 = time.time()

            try:
                size_mb = create_model(disease_key, arch_key, disease_info["classes"])
                elapsed = time.time() - t0
                log.info(f"  [DONE]  {arch_key:15s}  {size_mb:.1f} MB  ({elapsed:.1f}s)")
                created += 1
                total_size += size_mb
            except Exception as e:
                log.error(f"  [FAIL]  {arch_key:15s}  {str(e)}")

        print()

    elapsed_total = time.time() - start_time

    print("=" * 60)
    log.info(f"Setup complete in {elapsed_total:.0f}s")
    log.info(f"  Created: {created} models")
    log.info(f"  Skipped: {skipped} models (already existed)")
    log.info(f"  Total size: {total_size:.0f} MB")
    print("=" * 60)
    print()

    print_summary()


def print_summary():
    """Print final status of all model files."""
    print()
    print("Model Status:")
    print("-" * 50)
    for disease_key, disease_info in DISEASES.items():
        print(f"  {disease_info['name']}:")
        for arch_key in ARCHITECTURES:
            path = os.path.join(MODELS_DIR, disease_key, f"{arch_key}.keras")
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"    [OK]  {arch_key:15s}  {size:6.1f} MB")
            else:
                print(f"    [--]  {arch_key:15s}  MISSING")
    print()
    print("Run the app:  python server.py")
    print("Open browser: http://localhost:5000")
    print()


if __name__ == "__main__":
    main()
