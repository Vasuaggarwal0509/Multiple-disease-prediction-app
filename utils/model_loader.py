import os
import pickle
import json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Cache loaded models to avoid reloading
_model_cache = {}


def get_disease_config(disease_key):
    with open(os.path.join(CONFIG_DIR, "diseases.json")) as f:
        config = json.load(f)
    return config["diseases"].get(disease_key)


def load_pickle_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_keras_model(path):
    # Lazy import to avoid loading TF at startup unless needed
    from tensorflow.keras.models import load_model
    return load_model(path)


def load_model(disease_key, model_key):
    """Load a single model by disease and model key. Returns None if file missing."""
    cache_key = f"{disease_key}/{model_key}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    config = get_disease_config(disease_key)
    if not config:
        return None

    model_config = config["models"].get(model_key)
    if not model_config:
        return None

    model_path = os.path.join(BASE_DIR, model_config["file"])
    if not os.path.exists(model_path):
        return None

    fmt = model_config.get("format", "pickle")
    if fmt == "pickle":
        model = load_pickle_model(model_path)
    elif fmt == "keras":
        model = load_keras_model(model_path)
    else:
        return None

    _model_cache[cache_key] = model
    return model


def load_all_models(disease_key):
    """Load all available models for a disease. Returns dict of model_key -> model."""
    config = get_disease_config(disease_key)
    if not config:
        return {}

    models = {}
    for model_key in config["models"]:
        model = load_model(disease_key, model_key)
        if model is not None:
            models[model_key] = model
    return models


def list_available_models(disease_key):
    """List model keys that have actual files on disk."""
    config = get_disease_config(disease_key)
    if not config:
        return []

    available = []
    for model_key, model_config in config["models"].items():
        path = os.path.join(BASE_DIR, model_config["file"])
        available.append({
            "key": model_key,
            "name": model_config["name"],
            "available": os.path.exists(path),
            "format": model_config.get("format", "pickle"),
        })
    return available
