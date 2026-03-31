import os
import json
from flask import Flask, render_template, request, jsonify
from utils.metrics_loader import (
    load_disease_config,
    load_references,
    load_all_metrics,
    get_comparison_table,
)
from utils.model_loader import list_available_models
from utils.prediction import predict_tabular, predict_image_all_models

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB upload limit

BASE_DIR = os.path.dirname(__file__)

# Load configs once at startup
DISEASES = load_disease_config()
REFERENCES = load_references()


@app.route("/")
def index():
    """Landing page showing all disease cards."""
    return render_template("index.html", diseases=DISEASES)


@app.route("/disease/<disease_key>")
def disease_page(disease_key):
    """Individual disease page with prediction form and model comparison."""
    if disease_key not in DISEASES:
        return render_template("index.html", diseases=DISEASES, error="Disease not found"), 404

    disease = DISEASES[disease_key]
    models = list_available_models(disease_key)
    metrics = load_all_metrics(disease_key)
    comparison_df = get_comparison_table(disease_key)
    refs = REFERENCES.get(disease_key, {})

    return render_template(
        "disease.html",
        disease_key=disease_key,
        disease=disease,
        models=models,
        metrics=metrics,
        comparison_table=comparison_df.to_dict("records") if not comparison_df.empty else [],
        references=refs,
    )


@app.route("/predict/tabular", methods=["POST"])
def predict_tabular_route():
    """API endpoint for tabular disease prediction."""
    data = request.get_json()
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


@app.route("/predict/image", methods=["POST"])
def predict_image_route():
    """API endpoint for image-based disease prediction (all models)."""
    disease_key = request.form.get("disease_key")
    if disease_key not in DISEASES:
        return jsonify({"error": "Invalid disease"}), 400
    if DISEASES[disease_key]["type"] != "image":
        return jsonify({"error": "Not an image disease"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    results = predict_image_all_models(disease_key, image_file)
    return jsonify({"results": results, "disease": DISEASES[disease_key]["name"]})


@app.route("/comparison")
def comparison_overview():
    """Cross-disease model comparison dashboard."""
    all_comparisons = {}
    for disease_key, disease in DISEASES.items():
        if disease["type"] == "image":
            metrics = load_all_metrics(disease_key)
            if metrics:
                all_comparisons[disease_key] = {
                    "name": disease["name"],
                    "metrics": metrics,
                }
    return render_template("comparison.html", comparisons=all_comparisons)


@app.route("/methodology")
def methodology():
    """Research methodology and references page."""
    return render_template("methodology.html", diseases=DISEASES, references=REFERENCES)


@app.route("/api/metrics/<disease_key>")
def api_metrics(disease_key):
    """JSON API for fetching metrics (used by Chart.js)."""
    metrics = load_all_metrics(disease_key)
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
