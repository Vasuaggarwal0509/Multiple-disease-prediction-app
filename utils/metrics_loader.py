import json
import os
import logging
import pandas as pd

log = logging.getLogger(__name__)

METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def load_disease_config():
    path = os.path.join(CONFIG_DIR, "diseases.json")
    try:
        with open(path) as f:
            return json.load(f)["diseases"]
    except FileNotFoundError:
        log.error(f"Config file not found: {path}")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        log.error(f"Invalid config file {path}: {e}")
        return {}


def load_references():
    path = os.path.join(CONFIG_DIR, "references.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"References file not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        log.error(f"Invalid references file {path}: {e}")
        return {}


def load_model_metrics(disease_key, model_key):
    """Load pre-computed metrics JSON for a specific disease+model."""
    path = os.path.join(METRICS_DIR, disease_key, f"{model_key}_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log.error(f"Invalid metrics file {path}: {e}")
        return None


def load_all_metrics(disease_key):
    """Load metrics for all models of a given disease."""
    disease_dir = os.path.join(METRICS_DIR, disease_key)
    if not os.path.exists(disease_dir):
        return {}
    metrics = {}
    for fname in os.listdir(disease_dir):
        if fname.endswith("_metrics.json"):
            model_key = fname.replace("_metrics.json", "")
            try:
                with open(os.path.join(disease_dir, fname)) as f:
                    metrics[model_key] = json.load(f)
            except json.JSONDecodeError as e:
                log.error(f"Invalid metrics file {fname}: {e}")
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
