"""
Save as src/inference/test.py (replaces your layer-listing test).
Run:  python -m src.inference.test "C:\path\to\photo.jpg"
"""
import sys
import tensorflow as tf

from src.preprocessing.trainpre import preprocess_image
from src.dataset.attributemanage import SELECTED_ATTRIBUTES
from src.inference.stain_location import build_gradcam_model, analyse_image

MODEL_PATH = "project_model_best.keras"

if len(sys.argv) != 2:
    print('Usage: python -m src.inference.test "path_to_image"')
    sys.exit()

model = tf.keras.models.load_model(MODEL_PATH)
gradcam_model = build_gradcam_model(model)

image = tf.expand_dims(preprocess_image(sys.argv[1]), axis=0)
result = analyse_image(model, gradcam_model, image, SELECTED_ATTRIBUTES)

print("\nResult:", result["message"])
print("Label:", result["label"], f"({result['confidence'] * 100:.1f}%)")
print("Zone:", result["zone"])
print("Strongest heat cells (row, col, value):")
import numpy as np
heat = np.array(result["heatmap"])
for flat in np.argsort(heat.ravel())[::-1][:5]:
    r, c = divmod(int(flat), heat.shape[1])
    print(f"  {r:2d} {c:2d}  {heat[r, c]:.2f}")