import os
import logging
from flask import Flask, render_template, request, jsonify
from utils.metrics_loader import load_disease_config, load_references
from utils.prediction import predict_tabular

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB upload limit

BASE_DIR = os.path.dirname(__file__)

# Load configs once at startup
try:
    DISEASES = load_disease_config()
    REFERENCES = load_references()
    log.info(f"Loaded {len(DISEASES)} diseases, {len(REFERENCES)} reference entries")
except Exception as e:
    log.error(f"Failed to load config files: {e}")
    DISEASES = {}
    REFERENCES = {}


@app.route("/")
def index():
    """Landing page showing all disease cards."""
    return render_template("index.html", diseases=DISEASES)


@app.route("/disease/<disease_key>")
def disease_page(disease_key):
    """Individual disease page with single-model prediction interface."""
    if disease_key not in DISEASES:
        return render_template("index.html", diseases=DISEASES, error="Disease not found"), 404

    disease = DISEASES[disease_key]
    refs = REFERENCES.get(disease_key, {})

    return render_template(
        "disease.html",
        disease_key=disease_key,
        disease=disease,
        references=refs,
    )


@app.route("/predict/tabular", methods=["POST"])
def predict_tabular_route():
    """API endpoint for tabular disease prediction."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request - JSON required"}), 400
    disease_key = data.get("disease_key")
    values = data.get("values", [])

    if disease_key not in DISEASES:
        return jsonify({"error": "Invalid disease"}), 400
    if DISEASES[disease_key]["type"] != "tabular":
        return jsonify({"error": "Not a tabular disease"}), 400

    result = predict_tabular(disease_key, values)
    return jsonify(result)


# ───────────────────────────────────────────────────────────────
# Trained image model loaders (lazy)
# ───────────────────────────────────────────────────────────────

_eye_classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
_pneumonia_classes = ["Normal", "Pneumonia"]
_malaria_classes = ["Parasitized", "Uninfected"]


def _load_keras_by_disease(disease_key, cache_ref):
    """Load a trained keras model using path from config. Tries fallback_file if primary fails."""
    if cache_ref[0] is not None:
        return cache_ref[0]
    model_cfg = DISEASES.get(disease_key, {}).get("models", {}).get("trained", {})
    candidates = []
    if model_cfg.get("file"):
        candidates.append(os.path.join(BASE_DIR, model_cfg["file"]))
    if model_cfg.get("fallback_file"):
        candidates.append(os.path.join(BASE_DIR, model_cfg["fallback_file"]))
    if not candidates:
        log.error(f"No model path configured for {disease_key}")
        return None

    from tensorflow.keras.models import load_model
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            cache_ref[0] = load_model(path, compile=False)
            log.info(f"Loaded {disease_key} model: {path}")
            return cache_ref[0]
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")
    log.error(f"No {disease_key} model could be loaded")
    return None


_eye_cache = [None]
_pneumonia_cache = [None]
_malaria_cache = [None]


def _load_eye_model():
    return _load_keras_by_disease("eye_disease", _eye_cache)


def _load_pneumonia_model():
    return _load_keras_by_disease("pneumonia", _pneumonia_cache)


def _load_malaria_model():
    return _load_keras_by_disease("malaria", _malaria_cache)


# ───────────────────────────────────────────────────────────────
# Image prediction routes (single trained model each)
# ───────────────────────────────────────────────────────────────

def _get_image_file():
    """Extract uploaded image file with validation."""
    if "image" not in request.files:
        return None, (jsonify({"error": "No image uploaded"}), 400)
    f = request.files["image"]
    if not f.filename:
        return None, (jsonify({"error": "Empty file uploaded"}), 400)
    return f, None


@app.route("/predict/image/eye_disease", methods=["POST"])
def predict_eye_route():
    """OCT retinal disease inference using fine-tuned MobileNet V3."""
    image_file, err = _get_image_file()
    if err:
        return err
    model = _load_eye_model()
    if model is None:
        return jsonify({"error": "Trained model not found"}), 500
    try:
        import numpy as np
        from PIL import Image
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        img = Image.open(image_file).convert("RGB").resize((224, 224))
        x = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        return jsonify({
            "model_name": "MobileNet V3 (fine-tuned)",
            "label": _eye_classes[pred_class],
            "confidence": confidence,
            "probabilities": {_eye_classes[i]: float(preds[0][i]) for i in range(len(_eye_classes))},
        })
    except Exception as e:
        log.error(f"Eye prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/image/pneumonia", methods=["POST"])
def predict_pneumonia_route():
    """Chest X-ray pneumonia detection using custom CNN."""
    image_file, err = _get_image_file()
    if err:
        return err
    model = _load_pneumonia_model()
    if model is None:
        return jsonify({"error": "Trained model not found"}), 500
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(image_file).convert("RGB").resize((300, 300))
        x = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        preds = model.predict(x, verbose=0)
        prob = float(preds[0][0])
        pred_class = 1 if prob >= 0.5 else 0
        confidence = prob if pred_class == 1 else (1.0 - prob)
        return jsonify({
            "model_name": "Custom CNN (binary classifier)",
            "label": _pneumonia_classes[pred_class],
            "confidence": confidence,
            "probabilities": {"Normal": 1.0 - prob, "Pneumonia": prob},
        })
    except Exception as e:
        log.error(f"Pneumonia prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/image/malaria", methods=["POST"])
def predict_malaria_route():
    """Blood cell malaria parasite detection using custom CNN."""
    image_file, err = _get_image_file()
    if err:
        return err
    model = _load_malaria_model()
    if model is None:
        return jsonify({"error": "Trained model not found"}), 500
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(image_file).convert("RGB").resize((130, 130))
        x = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        preds = model.predict(x, verbose=0)
        prob = float(preds[0][0])
        pred_class = 1 if prob >= 0.5 else 0
        confidence = prob if pred_class == 1 else (1.0 - prob)
        return jsonify({
            "model_name": "Custom CNN (binary classifier)",
            "label": _malaria_classes[pred_class],
            "confidence": confidence,
            "probabilities": {"Parasitized": 1.0 - prob, "Uninfected": prob},
        })
    except Exception as e:
        log.error(f"Malaria prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/methodology")
def methodology():
    """Inference methodology page — describes each model's architecture and pipeline."""
    return render_template("methodology.html", diseases=DISEASES, references=REFERENCES)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
