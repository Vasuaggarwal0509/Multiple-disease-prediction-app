import json
import os
import pandas as pd

METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def load_disease_config():
    with open(os.path.join(CONFIG_DIR, "diseases.json")) as f:
        return json.load(f)["diseases"]


def load_references():
    with open(os.path.join(CONFIG_DIR, "references.json")) as f:
        return json.load(f)


def load_model_metrics(disease_key, model_key):
    """Load pre-computed metrics JSON for a specific disease+model."""
    path = os.path.join(METRICS_DIR, disease_key, f"{model_key}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_metrics(disease_key):
    """Load metrics for all models of a given disease."""
    disease_dir = os.path.join(METRICS_DIR, disease_key)
    if not os.path.exists(disease_dir):
        return {}
    metrics = {}
    for fname in os.listdir(disease_dir):
        if fname.endswith("_metrics.json"):
            model_key = fname.replace("_metrics.json", "")
            with open(os.path.join(disease_dir, fname)) as f:
                metrics[model_key] = json.load(f)
    return metrics


def get_comparison_table(disease_key):
    """Return a DataFrame comparing all models for a disease."""
    all_metrics = load_all_metrics(disease_key)
    if not all_metrics:
        return pd.DataFrame()

    rows = []
    for model_key, data in all_metrics.items():
        m = data.get("metrics", {})
        rows.append({
            "Model": data.get("model_name", model_key),
            "Accuracy": m.get("accuracy", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
            "F1 Score": m.get("f1_score", 0),
            "AUC-ROC": m.get("auc_roc", 0),
            "Parameters": data.get("parameters", "N/A"),
            "Training Time": data.get("training_time", "N/A"),
        })
    return pd.DataFrame(rows)
