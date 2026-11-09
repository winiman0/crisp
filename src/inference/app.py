"""
Flask backend for Detector UI.
http://127.0.0.1:5000
"""

import os
import tempfile

import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

from src.preprocessing.trainpre2 import preprocess_image
from src.dataset.attributemanage import SELECTED_ATTRIBUTES
from src.inference.stain_location import build_gradcam_model, analyse_image

MODEL_PATH = "project_model_best.keras"
UI_FILE = "detector.html"

app = Flask(__name__)

# Load the model ONCE when the server starts, not on every request.
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
gradcam_model = build_gradcam_model(model)
print("Model ready.")


@app.route("/")
def serve_ui():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), UI_FILE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    image_file = request.files["image"]

    # preprocess_image expects a file path, so save the upload to a temp file.
    suffix = os.path.splitext(image_file.filename or "")[1] or ".jpg"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            image_file.save(temp)
            temp_path = temp.name

        image = preprocess_image(temp_path)
        image_batch = tf.expand_dims(image, axis=0)   # (1, H, W, 3)

        result = analyse_image(model, gradcam_model, image_batch, SELECTED_ATTRIBUTES)
    except Exception as error:
        return jsonify({"error": f"Could not analyse the image: {error}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({
        "status": result["label"],                       # "clean" | "stain" | "defect"
        "label": result["label"].capitalize(),
        "garment": "Garment",                            # no garment-type output exists
        "confidence": round(result["confidence"] * 100),
        "message": result["message"],                    # the sentence to show and speak
        "zone": result["zone"]["zone"],                  # e.g. "lower left" or None
        "heatmap": result["heatmap"],                    # 14x14 list for an overlay
    })


if __name__ == "__main__":
    # use_reloader=False stops Flask loading the model twice in debug mode
    app.run(debug=True, use_reloader=False)