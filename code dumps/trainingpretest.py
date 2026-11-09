"""
Training Image Preprocessing

Image preprocessing:
    1. Load image
    2. Convert to RGB
    3. Resize to 224 x 224
    4. Normalize pixel values to 0-1

Training augmentation:
    - horizontal flip
    - brightness adjustment
    - contrast adjustment

"""

import os
import random
import tensorflow as tf

# Reuse the attribute definitions that you already validated.
from src.dataset.attributemanage import (
    ATTRIBUTE_DEFINITION_FILE,
    IMAGE_ATTRIBUTE_FILE,
    IMAGE_ROOT,
    SELECTED_ATTRIBUTES,
    read_attribute_definitions,
    find_selected_attributes,
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "clean",
    "stain",
    "defect"
]


# ============================================================
# 1. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Loads and prepares one image for the CNN.

    Steps:
        1. Read image
        2. Decode image as RGB
        3. Resize to 224 x 224
        4. Normalize pixel values to 0-1
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3]) 

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = image / 255.0

    return image


# ============================================================
# 2. DATA AUGMENTATION
# ============================================================

def augment_image(image):
    """
    Applies simple augmentation to training images.

    These changes help the model handle normal differences
    in image orientation and lighting.
    """

    # Horizontal flipping
    image = tf.image.random_flip_left_right(image)

    # Small brightness variation
    image = tf.image.random_brightness(
        image,
        max_delta=0.15
    )

    # Small contrast variation
    image = tf.image.random_contrast(
        image,
        lower=0.85,
        upper=1.15
    )

    return image


# ============================================================
# 3. FABRIC CONDITION DATASET
# ============================================================

def list_condition_images(root_dir):
    """
    Reads the fabric condition dataset.

    """

    filepaths = []
    labels = []

    for label_index, class_name in enumerate(CLASS_NAMES):

        class_folder = os.path.join(
            root_dir,
            class_name
        )

        if not os.path.exists(class_folder):
            print("Folder not found:", class_folder)
            continue

        for filename in os.listdir(class_folder):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):

                filepath = os.path.join(
                    class_folder,
                    filename
                )

                filepaths.append(filepath)
                labels.append(label_index)

    return filepaths, labels


# ============================================================
# 4. DEEPFASHION ATTRIBUTE DATASET
# ============================================================

def build_attribute_records():
    """
    Reads DeepFashion image annotations and creates:

        image path + 26 attribute labels

    Label meaning used here:

        1  = attribute present
        0  = attribute absent or unknown
    """

    # Read all 1,000 attribute definitions
    all_attributes = read_attribute_definitions(
        ATTRIBUTE_DEFINITION_FILE
    )

    # Find the positions of our selected 26 attributes
    attribute_positions = find_selected_attributes(
        all_attributes,
        SELECTED_ATTRIBUTES
    )

    positions = list(
        attribute_positions.values()
    )

    records = []

    with open(
        IMAGE_ATTRIBUTE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        # Skip image count
        file.readline()

        # Skip header
        file.readline()

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            image_path = parts[0]
            labels = parts[1:]

            # Create the 26-label vector
            label_vector = []

            for position in positions:

                if labels[position] == "1":
                    label_vector.append(1.0)
                else:
                    label_vector.append(0.0)

            full_path = os.path.join(
                IMAGE_ROOT,
                image_path
            )

            records.append(
                (full_path, label_vector)
            )

    return records


# ============================================================
# 5. AUGMENTATION + PREPROCESSING FOR CONDITION DATA
# ============================================================

def prepare_condition_image(
    image_path,
    label,
    training=True
):
    """
    Preprocesses one fabric-condition image.

    Returns:
        image, condition label
    """

    image = preprocess_image(
        image_path
    )

    if training:
        image = augment_image(
            image
        )

    return image, label


# ============================================================
# 6. AUGMENTATION + PREPROCESSING FOR ATTRIBUTE DATA
# ============================================================

def prepare_attribute_image(
    image_path,
    label_vector,
    training=True
):
    """
    Preprocesses one DeepFashion image.

    Returns:
        image, 26-attribute label vector
    """

    image = preprocess_image(
        image_path
    )

    if training:
        image = augment_image(
            image
        )

    return image, label_vector


# ============================================================
# 7. BUILD CONDITION DATASET
# ============================================================

def build_condition_dataset(
    filepaths,
    labels,
    batch_size=32,
    training=True
):
    """
    Builds the TensorFlow dataset for the
    fabric condition task.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    def process_image(
        image_path,
        label
    ):
        return prepare_condition_image(
            image_path,
            label,
            training
        )

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(filepaths)
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 8. BUILD ATTRIBUTE DATASET
# ============================================================

def build_attribute_dataset(
    records,
    batch_size=32,
    training=True
):
    """
    Builds the TensorFlow dataset for the
    DeepFashion attribute task.
    """

    filepaths = []
    label_vectors = []

    for image_path, label_vector in records:

        filepaths.append(
            image_path
        )

        label_vectors.append(
            label_vector
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            filepaths,
            label_vectors
        )
    )

    def process_image(
        image_path,
        label_vector
    ):
        return prepare_attribute_image(
            image_path,
            label_vector,
            training
        )

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(filepaths)
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 9. SIMPLE TEST
# ============================================================
if __name__ == "__main__":

    print()
    print("Training Preprocessing Test")
    print("=" * 50)

    # --------------------------------------------------------
    # 1. Test basic settings
    # --------------------------------------------------------

    print()
    print("Image preprocessing settings:")
    print("  Image size :", IMG_SIZE)
    print("  Channels   : RGB")
    print("  Normalized : 0-1")

    # Test
    # Fabric Condition dataset 

    CONDITION_ROOT = r"C:\Users\User\crisp\dataset\relabeled_cond"

    print()
    print("Testing fabric condition dataset")
    
    condition_files, condition_labels = list_condition_images(
        CONDITION_ROOT
    )

    print("Images found:", len(condition_files))
    print("Labels found:", len(condition_labels))

    if len(condition_files) == 0:
        print("ERROR: No fabric condition images found.")
    else:
        print("Fabric condition dataset: OK")

        # Test one image
        image = preprocess_image(
            condition_files[0]
        )

        print("Sample image shape:", image.shape)
        print(
            "Pixel range:",
            float(tf.reduce_min(image)),
            "to",
            float(tf.reduce_max(image))
        )

    # DeepFashion 
    print()
    print("Testing DeepFashion dataset")

    attribute_records = build_attribute_records()

    print("Images found:", len(attribute_records))

    if len(attribute_records) == 0:
        print("ERROR: No DeepFashion records found.")
    else:
        print("DeepFashion dataset: OK")

        sample_path, sample_labels = attribute_records[0]

        print("Sample image:", sample_path)
        print("Label vector length:", len(sample_labels))

        # Test one DeepFashion image
        image = preprocess_image(
            sample_path
        )

        print("Sample image shape:", image.shape)

        print(
            "Pixel range:",
            float(tf.reduce_min(image)),
            "to",
            float(tf.reduce_max(image))
        )

    print()
    print("=" * 50)
    print("Preprocessing test completed.")