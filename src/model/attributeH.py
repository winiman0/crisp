from tensorflow.keras import layers

from src.dataset.attributemanage import SELECTED_ATTRIBUTES


# The head's output size follows the attribute list automatically
NUM_ATTRIBUTES = len(SELECTED_ATTRIBUTES)


def attribute_head(spatial_features):
    """
    Takes the SPATIAL feature map (the 7 x 7 grid of features, not yet
    averaged into one vector).

    Why: buttons, zippers, pockets and collars are small details. A single
    averaged vector for the whole garment washes them out, which is the same
    problem the condition head had with small stains. This head works on the
    grid first and only then summarises it.

    It summarises the grid two ways and uses both:
        average over the grid -> "how much of the garment looks like this"
                                 (good for colours and patterns)
        maximum over the grid -> "does any spot look like this"
                                 (good for small parts: a button, a zipper)
    """

    # Two conv layers that work on the grid. No MaxPooling here, so the
    # 7 x 7 grid keeps its detail.
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="attribute_conv_1"
    )(spatial_features)
    x = layers.BatchNormalization(momentum=0.9)(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="attribute_conv_2"
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)

    # Summarise the grid both ways, then join the two summaries
    average_summary = layers.GlobalAveragePooling2D()(x)
    maximum_summary = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([average_summary, maximum_summary])

    x = layers.Dense(
        128,
        activation="relu",
        name="attribute_dense"
    )(x)

    x = layers.Dropout(
        0.3
    )(x)

    attribute_output = layers.Dense(
        NUM_ATTRIBUTES,
        activation="sigmoid",
        name="attribute_output"
    )(x)

    return attribute_output


# Same function under the other name, in case model.py imports this one
build_attribute_head = attribute_head