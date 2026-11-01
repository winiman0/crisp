"""
Flask backend for the Stain & Defect Detector UI.

Run it with:   python app.py
Then open:     http://127.0.0.1:5000

This serves your HTML page AND handles the /predict endpoint, so
everything lives on one address (no cross-origin issues).
"""

from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# The HTML file should sit in the same folder as this script.
UI_FILE = "stain-defect-detector-ui.html"


@app.route("/")
def serve_ui():
    """Serves the UI page itself when you visit http://127.0.0.1:5000"""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), UI_FILE)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives the uploaded photo from the browser, runs it through the
    trained CNN, and returns the result as JSON.

    Right now this returns a placeholder result — replace the marked
    section below with your actual model.predict() call once your
    trained model (project_model_best.keras) is ready.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    image_file = request.files["image"]

    # -----------------------------------------------------------------
    # TODO: replace this placeholder block with real inference:
    #
    #   from tensorflow.keras.models import load_model
    #   model = load_model("project_model_best.keras")  # load once, outside predict()
    #
    #   image = preprocess_image(image_file)  # reuse your trainpre.py preprocessing
    #   outputs = model(image, training=False)
    #   condition_pred = outputs["condition_output"]
    #   ... turn condition_pred into a label + confidence ...
    # -----------------------------------------------------------------

    result = {
        "status": "clean",       # "clean" | "stain" | "defect"
        "label": "Clean",
        "garment": "Shirt",
        "confidence": 98
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)