"""
stain_location.py   ->  save as  src/inference/stain_location.py

Tells the user WHERE on the garment the model sees the stain or defect.

How it works (plain English):
  1. Run the image through the trained model.
  2. Ask: "which cells of the 7x7 feature map pushed the Stain score up?"
     This is Grad-CAM: take the gradient of the class score with respect to
     the feature map, average it per channel to get channel weights, then
     add up the channels weighted by those numbers.
  3. Keep only the strong cells, then see which part of the garment they sit in
     (upper / middle / lower  x  left / centre / right).
  4. Turn that into a sentence the app can speak.

Everything is written by hand with tf.GradientTape and numpy.
No pretrained weights and no ready-made Grad-CAM function are used.

Method reference: Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV.
"""

import numpy as np

CONDITION_CLASSES = ["clean", "stain", "defect"]

# Attributes that showed a real gap in evaluation, with the threshold chosen on
# the validation half. Everything else is NOT spoken (the model did not learn it).
RELIABLE_ATTRIBUTES = {
    "red": 0.10,
    "floral": 0.30,
    "striped": 0.40,
    "denim": 0.15,
    "leather": 0.15,
}

# A cell counts as "part of the stain" if its heat is at least this fraction
# of the hottest cell.
HEAT_CUTOFF = 0.5

# Row / column names for the 3 x 3 garment grid (viewed as the photo is shown).
ROW_NAMES = ["upper", "middle", "lower"]
COL_NAMES = ["left", "centre", "right"]


# ----------------------------------------------------------------------------
# Step A: find the last backbone feature layer inside the trained model
# ----------------------------------------------------------------------------
def find_feature_layer(model):
    """
    Walk the layers in order and remember the last layer with a 4-D output
    (batch, height, width, channels). Stop when the first head layer appears
    (a name starting with "attribute" or "condition"), so the answer is the
    backbone's last spatial layer, the one both heads read from.
    For this project that is re_lu_12, shape (None, 14, 14, 256).
    """
    chosen = None
    for layer in model.layers:
        name = layer.name.lower()
        if name.startswith("condition") or name.startswith("attribute"):
            break
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            chosen = layer
    if chosen is None:
        raise ValueError(
            "No spatial layer found before the heads. Run print_candidate_layers(model) "
            "and pass the right layer name as feature_layer_name."
        )
    return chosen


def print_candidate_layers(model):
    """Helper: lists every layer with a 4-D output so you can pick by eye."""
    for layer in model.layers:
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            print(f"{layer.name:35s} {tuple(shape)}")


# ----------------------------------------------------------------------------
# Step B: Grad-CAM on the backbone feature map (14x14) (needs TensorFlow, imported only here)
# ----------------------------------------------------------------------------
def build_gradcam_model(model, feature_layer_name=None):
    """Makes a small model that returns (feature map, condition scores)."""
    import tensorflow as tf

    if feature_layer_name is None:
        feature_layer = find_feature_layer(model)
    else:
        feature_layer = model.get_layer(feature_layer_name)

    print(f"Grad-CAM reads feature layer: {feature_layer.name} {tuple(feature_layer.output.shape)}")

    return tf.keras.Model(
        inputs=model.input,
        outputs=[feature_layer.output, model.output["condition_output"]],
    )


def compute_heatmap(gradcam_model, image_batch, class_index=None):
    """
    image_batch: shape (1, H, W, 3), preprocessed EXACTLY like at training time.
    Returns (heatmap (grid x grid) scaled 0..1, condition probabilities, class used).
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        # training=False -> batch norm uses stored statistics, dropout is off
        feature_map, scores = gradcam_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(scores[0]))
        class_score = scores[:, class_index]

    # How much does each feature-map value change the class score?
    gradients = tape.gradient(class_score, feature_map)          # (1, 7, 7, C)

    # One weight per channel = average gradient over the 7x7 positions
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))     # (1, C)

    # Weighted sum of channels, keep only positive evidence (ReLU)
    weighted = feature_map * channel_weights[:, tf.newaxis, tf.newaxis, :]
    heatmap = tf.nn.relu(tf.reduce_sum(weighted, axis=-1))[0].numpy()   # (7, 7)

    top = heatmap.max()
    if top > 0:
        heatmap = heatmap / top                                  # scale 0..1

    return heatmap, scores[0].numpy(), class_index


# ----------------------------------------------------------------------------
# Step C: heatmap -> garment zone (pure numpy, no TensorFlow needed)
# ----------------------------------------------------------------------------
def heatmap_to_zone(heatmap, cutoff=HEAT_CUTOFF):
    """
    Splits the 7x7 grid into 3 rows x 3 columns and adds up the heat in each
    zone, counting only strong cells. Returns the main zone and how
    concentrated the evidence is.
    """
    grid = heatmap.shape[0]
    strong = np.where(heatmap >= cutoff, heatmap, 0.0)

    if strong.sum() == 0:
        return {"zone": None, "row": None, "col": None,
                "concentration": 0.0, "spread": True}

    # Cut the grid into three nearly equal bands (7 -> 3,2,2 style splits)
    row_bands = np.array_split(np.arange(grid), 3)
    col_bands = np.array_split(np.arange(grid), 3)

    zone_heat = np.zeros((3, 3))
    for r, rows in enumerate(row_bands):
        for c, cols in enumerate(col_bands):
            zone_heat[r, c] = strong[np.ix_(rows, cols)].sum()

    best_r, best_c = np.unravel_index(np.argmax(zone_heat), zone_heat.shape)
    concentration = float(zone_heat[best_r, best_c] / zone_heat.sum())

    return {
        "zone": f"{ROW_NAMES[best_r]} {COL_NAMES[best_c]}",
        "row": ROW_NAMES[best_r],
        "col": COL_NAMES[best_c],
        "concentration": concentration,   # 1.0 = all in one zone
        # If half or less of the evidence is in one zone, say "spread out"
        "spread": concentration <= 0.5,
    }


# ----------------------------------------------------------------------------
# Step D: build the sentence for the app
# ----------------------------------------------------------------------------
# Each grid zone gets a garment-part phrase. This assumes the garment is
# photographed front-on and upright (the usual way a shopper holds it up),
# so the top centre is the neckline, the top corners are shoulders/sleeves,
# and the bottom is the hem. "left" / "right" are as seen in the photo.
ZONE_PHRASES = {
    ("upper", "left"):    "near the upper left, around the shoulder or sleeve",
    ("upper", "centre"):  "around the collar or upper chest",
    ("upper", "right"):   "near the upper right, around the shoulder or sleeve",
    ("middle", "left"):   "on the left side",
    ("middle", "centre"): "in the centre of the garment",
    ("middle", "right"):  "on the right side",
    ("lower", "left"):    "near the lower left, close to the hem",
    ("lower", "centre"):  "near the bottom hem",
    ("lower", "right"):   "near the lower right, close to the hem",
}


def describe_zone(zone_info):
    if zone_info["zone"] is None:
        return "in an unclear area"

    phrase = ZONE_PHRASES[(zone_info["row"], zone_info["col"])]

    # Evidence split over several zones: say so, but still name the main one
    if zone_info["spread"]:
        return f"in several areas, mostly {phrase}"
    return phrase


def build_message(condition_scores, zone_info, attribute_scores=None, attribute_names=None):
    """
    condition_scores : array of 3 probabilities (clean, stain, defect)
    zone_info        : result of heatmap_to_zone
    attribute_scores : array of attribute probabilities (optional)
    attribute_names  : list of attribute names in the same order
    """
    label = CONDITION_CLASSES[int(np.argmax(condition_scores))]
    confidence = float(np.max(condition_scores))

    if label == "clean":
        message = "No stain or defect detected."
    else:
        message = f"{label.capitalize()} detected {describe_zone(zone_info)}."

    # Only mention attributes the model actually learned, and hedge them
    described = []
    if attribute_scores is not None and attribute_names is not None:
        for name, cutoff in RELIABLE_ATTRIBUTES.items():
            if name in attribute_names:
                if attribute_scores[attribute_names.index(name)] >= cutoff:
                    described.append(name)
    if described:
        message += " The garment may be " + ", ".join(described) + "."

    return {"label": label, "confidence": confidence, "message": message}


# ----------------------------------------------------------------------------
# One call for the app
# ----------------------------------------------------------------------------
def analyse_image(model, gradcam_model, image_batch, attribute_names):
    """
    Returns everything /predict needs:
      label, confidence, message, zone details, 7x7 heatmap for the overlay.
    """
    outputs = model(image_batch, training=False)
    attribute_scores = outputs["attribute_output"][0].numpy()

    heatmap, condition_scores, _ = compute_heatmap(gradcam_model, image_batch)
    label_index = int(np.argmax(condition_scores))

    # Only locate things when something was found
    if CONDITION_CLASSES[label_index] == "clean":
        zone_info = {"zone": None, "row": None, "col": None,
                     "concentration": 0.0, "spread": True}
        heatmap = np.zeros_like(heatmap)
    else:
        # Re-run Grad-CAM for the predicted class so the map explains that class
        heatmap, _, _ = compute_heatmap(gradcam_model, image_batch, class_index=label_index)
        zone_info = heatmap_to_zone(heatmap)

    result = build_message(condition_scores, zone_info, attribute_scores, attribute_names)
    result["zone"] = zone_info
    result["heatmap"] = heatmap.tolist()
    return result