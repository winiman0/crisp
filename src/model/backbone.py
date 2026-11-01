import tensorflow as tf
from tensorflow.keras import layers


def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x

    y = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=stride, padding="same")(x)
    y = layers.BatchNormalization(momentum=0.9)(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same")(y)
    y = layers.BatchNormalization(momentum=0.9)(y)

    if downsample or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9)(shortcut)

    x = layers.Add()([y, shortcut])
    x = layers.ReLU()(x)
    return x


def build_backbone(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape, name="image_input")

    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    # x here is the SPATIAL feature map, e.g. shape (batch, 7, 7, 256) for 224x224 input.
    # This still has "where in the image" information — don't collapse it yet.

    spatial_features = x  # <-- NEW: kept for the condition head

    shared_features = layers.GlobalAveragePooling2D(
        name="shared_feature_vector"
    )(x)  # <-- same as before, used by the attribute head

    return tf.keras.Model(
        inputs=inputs,
        outputs=[spatial_features, shared_features],
        name="custom_cnn_backbone"
    )