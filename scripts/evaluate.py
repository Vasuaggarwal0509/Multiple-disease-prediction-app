"""
Evaluate deployed image models against labeled test images.

Workflow (single script):
  1. Scan test_data/<disease>/<class>/*.jpg|png to build ground truth
  2. Write test_data/ground_truth.csv for reference / UI use
  3. Load each trained model from ml_models/
  4. Run inference on every test image
  5. Compute per-disease metrics: accuracy, precision, recall, F1, AUC-ROC,
     confusion matrix, per-class stats
  6. Save test_data/evaluation_results.json

Usage:
    python scripts/evaluate.py
"""

import os
import sys
import csv
import json
import logging
from collections import Counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "test_data")
GROUND_TRUTH_CSV = os.path.join(TEST_DIR, "ground_truth.csv")
OUTPUT_JSON = os.path.join(TEST_DIR, "evaluation_results.json")

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

CLASS_LABELS = {
    "eye_disease": ["CNV", "DME", "DRUSEN", "NORMAL"],
    "pneumonia": ["Normal", "Pneumonia"],
    "malaria": ["Parasitized", "Uninfected"],
}

MODEL_PATHS = {
    "eye_disease": ["ml_models/eye_disease/trained_model.keras",
                    "ml_models/eye_disease/trained_model.h5"],
    "pneumonia": ["ml_models/pneumonia/trained_model.h5"],
    "malaria": ["ml_models/malaria/trained_model.h5"],
}


# ── Ground-truth scanner ──
def scan_test_data():
    """Walk test_data/<disease>/<class>/* and build list of {disease, filename, true_label}."""
    rows = []
    if not os.path.isdir(TEST_DIR):
        log.error(f"{TEST_DIR} does not exist")
        return rows

    for disease in sorted(os.listdir(TEST_DIR)):
        disease_path = os.path.join(TEST_DIR, disease)
        if not os.path.isdir(disease_path):
            continue
        for cls in sorted(os.listdir(disease_path)):
            cls_path = os.path.join(disease_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for img in sorted(os.listdir(cls_path)):
                if img.lower().endswith(VALID_EXT):
                    rel = os.path.relpath(os.path.join(cls_path, img), BASE_DIR).replace("\\", "/")
                    rows.append({"disease": disease, "filename": rel, "true_label": cls})
    return rows


def write_csv(rows):
    with open(GROUND_TRUTH_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease", "filename", "true_label"])
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote ground truth: {GROUND_TRUTH_CSV} ({len(rows)} rows)")


# ── Model loading ──
def load_model(candidates):
    from tensorflow.keras.models import load_model as keras_load
    for rel in candidates:
        path = os.path.join(BASE_DIR, rel)
        if not os.path.exists(path):
            continue
        try:
            return keras_load(path, compile=False)
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")
    return None


# ── Preprocessing + prediction ──
def predict_probs(model, disease_key, img_path):
    import numpy as np
    from PIL import Image

    if disease_key == "eye_disease":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
        return model.predict(preprocess_input(x), verbose=0)[0]

    size = (300, 300) if disease_key == "pneumonia" else (130, 130)
    img = Image.open(img_path).convert("RGB").resize(size)
    x = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)
    p = float(model.predict(x, verbose=0)[0][0])
    return np.array([1.0 - p, p], dtype="float32")


# ── Metrics ──
def compute_metrics(disease_key, y_true, y_pred, y_proba):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
    )

    classes = CLASS_LABELS[disease_key]
    num_classes = len(classes)
    is_binary = (num_classes == 2)
    avg = "binary" if is_binary else "weighted"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    try:
        if is_binary:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        log.warning(f"AUC-ROC computation failed: {e}")
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls in classes:
        s = report.get(cls, {})
        per_class[cls] = {
            "precision": round(s.get("precision", 0.0), 4),
            "recall": round(s.get("recall", 0.0), 4),
            "f1": round(s.get("f1-score", 0.0), 4),
            "support": int(s.get("support", 0)),
        }

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc), 4) if not np.isnan(auc) else None,
        "confusion_matrix": cm.tolist(),
        "class_labels": classes,
        "per_class_metrics": per_class,
    }


def evaluate_disease(disease_key, rows, model):
    import numpy as np
    classes = CLASS_LABELS[disease_key]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    disease_rows = [r for r in rows if r["disease"] == disease_key]
    if not disease_rows:
        return None

    y_true, y_pred, y_proba = [], [], []

    for i, r in enumerate(disease_rows, 1):
        img_path = os.path.join(BASE_DIR, r["filename"])
        if not os.path.exists(img_path):
            log.warning(f"Missing: {img_path}")
            continue
        try:
            probs = predict_probs(model, disease_key, img_path)
        except Exception as e:
            log.error(f"Prediction failed for {img_path}: {e}")
            continue
        y_true.append(class_to_idx[r["true_label"]])
        y_pred.append(int(np.argmax(probs)))
        y_proba.append(probs)
        if i % 10 == 0:
            log.info(f"  {disease_key}: {i}/{len(disease_rows)} processed")

    if not y_true:
        return None

    return compute_metrics(
        disease_key,
        np.array(y_true),
        np.array(y_pred),
        np.vstack(y_proba),
    )


# ── Output formatting ──
def format_confusion_matrix(cm, classes):
    col_w = max(10, max(len(c) for c in classes) + 1)
    header = " " * col_w + "".join(f"{c:>{col_w}s}" for c in classes)
    lines = ["  " + header]
    for i, tr in enumerate(classes):
        row = f"{tr:<{col_w}s}" + "".join(f"{cm[i][j]:>{col_w}d}" for j in range(len(classes)))
        lines.append("  " + row)
    return "\n".join(lines)


def print_disease_result(disease_key, metrics):
    classes = CLASS_LABELS[disease_key]
    print(f"\n[{disease_key}]")
    print(f"  Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision  : {metrics['precision']*100:.2f}%")
    print(f"  Recall     : {metrics['recall']*100:.2f}%")
    print(f"  F1 Score   : {metrics['f1_score']*100:.2f}%")
    auc = metrics["auc_roc"]
    print(f"  AUC-ROC    : {auc:.4f}" if auc is not None else "  AUC-ROC    : N/A")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(format_confusion_matrix(metrics["confusion_matrix"], classes))

    print(f"\n  Per-class metrics:")
    col_w = max(12, max(len(c) for c in classes) + 2)
    print(f"    {'Class':<{col_w}s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for cls in classes:
        s = metrics["per_class_metrics"][cls]
        print(f"    {cls:<{col_w}s} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f} {s['support']:>10d}")


def main():
    print("=" * 70)
    print("  Disease Prediction System — Model Evaluation")
    print("=" * 70)

    # Step 1: Scan folders + build ground truth
    rows = scan_test_data()
    if not rows:
        log.error("No images in test_data/. Drop images into class subfolders and rerun.")
        sys.exit(1)

    log.info(f"Found {len(rows)} labeled test images")
    counts = Counter((r["disease"], r["true_label"]) for r in rows)
    for (disease, cls), count in sorted(counts.items()):
        log.info(f"  {disease}/{cls}: {count}")

    write_csv(rows)

    # Step 2: Evaluate each disease
    all_results = {}
    for disease_key in ["eye_disease", "pneumonia", "malaria"]:
        n_disease = sum(1 for r in rows if r["disease"] == disease_key)
        if n_disease == 0:
            log.info(f"\n[{disease_key}] No test images — skipping")
            continue

        log.info(f"\n[{disease_key}] Loading model...")
        model = load_model(MODEL_PATHS[disease_key])
        if model is None:
            log.error(f"[{disease_key}] Model failed to load — skipping")
            continue

        log.info(f"[{disease_key}] Running inference on {n_disease} images...")
        metrics = evaluate_disease(disease_key, rows, model)
        if metrics is None:
            continue

        all_results[disease_key] = metrics
        print_disease_result(disease_key, metrics)

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Disease':<15s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'AUC':>8s}")
    print("  " + "-" * 56)
    for dk, m in all_results.items():
        auc_str = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "N/A"
        print(f"  {dk:<15s} {m['accuracy']*100:>7.2f}% {m['precision']*100:>7.2f}% "
              f"{m['recall']*100:>7.2f}% {m['f1_score']*100:>7.2f}% {auc_str:>8s}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_JSON}")
    print(f"  Ground truth : {GROUND_TRUTH_CSV}")


if __name__ == "__main__":
    main()
