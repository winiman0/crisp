#from src.inference.stain_location import build_gradcam_model, print_candidate_layers
#import tensorflow as tf

#model = tf.keras.models.load_model("project_model_best.keras")
#gm = build_gradcam_model(model)

# src/inference/test.py  (temporary)
import tensorflow as tf

model = tf.keras.models.load_model("project_model_best.keras")

print("TOP-LEVEL LAYERS")
for layer in model.layers:
    try:
        shape = tuple(layer.output.shape)
    except Exception:
        shape = "?"
    print(f"{layer.name:35s} {type(layer).__name__:22s} {shape}")

print("\nNESTED")
for layer in model.layers:
    if hasattr(layer, "layers"):
        print("inside:", layer.name)
        for inner in layer.layers[-6:]:
            print("   ", inner.name, tuple(inner.output.shape))