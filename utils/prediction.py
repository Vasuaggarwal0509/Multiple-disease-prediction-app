import logging
import numpy as np
from utils.model_loader import load_model, get_disease_config
from utils.preprocessing import preprocess_image

log = logging.getLogger(__name__)


def predict_tabular(disease_key, input_values):
    """Run prediction on tabular data using the SVM model."""
    model = load_model(disease_key, "svm")
    if model is None:
        return {"error": "Model not found or failed to load"}

    config = get_disease_config(disease_key)
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


def predict_image_all_models(disease_key, image_file):
    """Run prediction on an image using all available models for a disease."""
    config = get_disease_config(disease_key)
    if not config:
        return [{"error": "Disease not configured"}]

    classes = config["dataset"]["classes"]
    results = []

    for model_key, model_config in config["models"].items():
        model = load_model(disease_key, model_key)
        if model is None:
            results.append({
                "model_key": model_key,
                "model_name": model_config["name"],
                "available": False,
                "error": "Model not found or failed to load",
            })
            continue

        try:
            preprocess_type = model_config.get("preprocess", "mobilenet_v3")
            image_file.seek(0)
            img = preprocess_image(image_file, preprocess_type)

            preds = model.predict(img, verbose=0)
            pred_class = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            probabilities = {classes[i]: float(preds[0][i]) for i in range(len(classes))}

            results.append({
                "model_key": model_key,
                "model_name": model_config["name"],
                "available": True,
                "prediction": pred_class,
                "label": classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}",
                "confidence": confidence,
                "probabilities": probabilities,
            })
        except Exception as e:
            log.error(f"Image prediction failed for {model_key}: {e}")
            results.append({
                "model_key": model_key,
                "model_name": model_config["name"],
                "available": True,
                "error": str(e),
            })

    return results
