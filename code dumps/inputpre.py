import tensorflow as tf

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)


# ============================================================
# INPUT PREPROCESSING
# ============================================================

def preprocess_input(image_path):
    """
    Prepares an image received from the web application
    before sending it to the trained CNN.

    Steps:
    1. Read image
    2. Decode as RGB
    3. Resize to 224 x 224
    4. Normalize pixel values to 0-1
    5. Add batch dimension
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = image / 255.0

    # CNN expects:
    # (batch, height, width, channels)
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Input Image Preprocessing")
    print("=" * 40)

    print("Image size:", IMG_SIZE)
    print("Normalization: 0-1")
    print("Batch dimension: added")