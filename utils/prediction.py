import logging
import os
import pickle
import json

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

_svm_cache = {}


def _get_disease_config(disease_key):
    try:
        with open(os.path.join(CONFIG_DIR, "diseases.json")) as f:
            return json.load(f)["diseases"].get(disease_key)
    except Exception as e:
        log.error(f"Failed to load disease config: {e}")
        return None


def _load_svm(disease_key):
    if disease_key in _svm_cache:
        return _svm_cache[disease_key]
    config = _get_disease_config(disease_key)
    if not config:
        return None
    model_rel_path = config.get("models", {}).get("svm", {}).get("file")
    if not model_rel_path:
        return None
    path = os.path.join(BASE_DIR, model_rel_path)
    if not os.path.exists(path):
        log.warning(f"SVM model not found: {path}")
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        _svm_cache[disease_key] = model
        return model
    except Exception as e:
        log.error(f"Failed to load SVM model {path}: {e}")
        return None


def predict_tabular(disease_key, input_values):
    """Run prediction on tabular data using the SVM model."""
    model = _load_svm(disease_key)
    if model is None:
        return {"error": "Model not found or failed to load"}

    config = _get_disease_config(disease_key)
    if not config:
        return {"error": "Disease not configured"}

    classes = config["dataset"]["classes"]
    expected_features = len(config.get("features", []))

    if expected_features and len(input_values) != expected_features:
        return {"error": f"Expected {expected_features} features, got {len(input_values)}"}

    try:
        input_array = [float(v) for v in input_values]
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid input values: {e}"}

    try:
        prediction = model.predict([input_array])
        pred_class = int(prediction[0])
    except Exception as e:
        log.error(f"Tabular prediction failed: {e}")
        return {"error": f"Prediction failed: {e}"}

    return {
        "prediction": pred_class,
        "label": classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}",
        "model": "Support Vector Machine",
    }
