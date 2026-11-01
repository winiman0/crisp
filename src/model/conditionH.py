from tensorflow.keras import layers

NUM_CONDITIONS = 3


def build_condition_head(spatial_features):
    # One extra conv stage, specialized for localized texture/defect cues
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(spatial_features)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)

    # Pool AFTER this head's own localized processing, not before
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu", name="condition_dense")(x)
    x = layers.Dropout(0.4)(x)  # slightly higher, given the overfitting signature in your val loss

    condition_output = layers.Dense(
        NUM_CONDITIONS, activation="softmax", name="condition_output"
    )(x)

    return condition_output