import numpy as np
from utils.model_loader import load_model, get_disease_config, load_all_models
from utils.preprocessing import preprocess_image


def predict_tabular(disease_key, input_values):
    """Run prediction on tabular data using the SVM model.

    Args:
        disease_key: 'diabetes', 'heart', or 'parkinsons'
        input_values: list of numeric values matching feature order

    Returns:
        dict with prediction result and class label
    """
    model = load_model(disease_key, "svm")
    if model is None:
        return {"error": "Model not found"}

    config = get_disease_config(disease_key)
    classes = config["dataset"]["classes"]

    input_array = [float(v) for v in input_values]
    prediction = model.predict([input_array])
    pred_class = int(prediction[0])

    return {
        "prediction": pred_class,
        "label": classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}",
        "model": "Support Vector Machine",
    }


def predict_image_all_models(disease_key, image_file):
    """Run prediction on an image using all available models for a disease.

    Args:
        disease_key: e.g. 'eye_disease', 'brain_tumor', 'pneumonia', 'malaria'
        image_file: file-like object

    Returns:
        list of dicts, one per model, with prediction and probabilities
    """
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
            })
            continue

        preprocess_type = model_config.get("preprocess", "mobilenet_v3")
        # Reset file pointer for each model
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
            "label": classes[pred_class],
            "confidence": confidence,
            "probabilities": probabilities,
        })

    return results
