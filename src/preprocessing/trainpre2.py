import os
import tensorflow as tf

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
BATCH_SIZE = 32

CONDITION_CLASSES = ["clean", "stain", "defect"]

CONDITION_DATASET = r"C:\Users\User\crisp\dataset\relabeled_cond"


# ============================================================
# BASIC IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess one image.

    Steps:
    1. Read image
    2. Decode as RGB
    3. Resize to 224 x 224
    4. Normalize pixel values to 0-1
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def augment_image(image):
    """
    Apply random augmentation during training.
    """

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(
        image,
        max_delta=0.15
    )

    image = tf.image.random_contrast(
        image,
        lower=0.85,
        upper=1.15
    )

    return image


# ============================================================
# FABRIC CONDITION DATASET
# ============================================================

def list_condition_images(root_dir):
    """
    Find all fabric condition images and their class labels.
    """

    image_paths = []
    labels = []

    for class_index, class_name in enumerate(CONDITION_CLASSES):

        class_dir = os.path.join(root_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: folder not found: {class_dir}")
            continue

        for filename in os.listdir(class_dir):

            file_path = os.path.join(class_dir, filename)

            if os.path.isfile(file_path):
                image_paths.append(file_path)
                labels.append(class_index)

    return image_paths, labels


def build_condition_dataset(
    image_paths,
    labels,
    training=True,
    batch_size=BATCH_SIZE
):
    """
    Build the TensorFlow dataset for fabric condition classification.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def load_condition_image(image_path, label):

        image = preprocess_image(image_path)

        if training:
            image = augment_image(image)

        return image, label

    dataset = dataset.map(
        load_condition_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths)
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# DEEPFASHION ATTRIBUTE RECORDS
# ============================================================

def build_attribute_records():
    """
    Read DeepFashion annotations and create:
        image path + 26 selected attribute labels
    """

    attribute_definitions = read_attribute_definitions(
        ATTRIBUTE_DEFINITION_FILE
    )

    # Convert selected attribute names into their positions in the
    # 1000-attribute list, walking SELECTED_ATTRIBUTES in its OWN order.
    # This guarantees that label column i is always SELECTED_ATTRIBUTES[i],
    # so the names printed in evaluate.py match the real columns.
    # (.index() raises an error if a name is missing, instead of silently
    # skipping it.)
    attribute_indices = []

    for attribute_name in SELECTED_ATTRIBUTES:

        occurrences = attribute_definitions.count(attribute_name)
        if occurrences > 1:
            print(
                f"Warning: '{attribute_name}' appears {occurrences} times "
                f"in the attribute list; only the first one is used."
            )

        index = attribute_definitions.index(attribute_name)
        attribute_indices.append(index)

    print(
        f"Selected attributes found: "
        f"{len(attribute_indices)}"
    )

    records = []

    with open(
        IMAGE_ATTRIBUTE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        image_count = int(file.readline().strip())

        # Skip attribute header
        file.readline()

        for line in file:

            parts = line.strip().split()

            if len(parts) < 1001:
                continue

            image_relative_path = parts[0]

            all_labels = parts[1:]

            selected_labels = []

            for index in attribute_indices:

                if all_labels[index] == "1":
                    selected_labels.append(1.0)
                else:
                    selected_labels.append(0.0)

            image_path = os.path.join(
                IMAGE_ROOT,
                image_relative_path
            )

            records.append(
                (
                    image_path,
                    selected_labels
                )
            )

    return records
# ============================================================
# DEEPFASHION ATTRIBUTE DATASET
# ============================================================

def build_attribute_dataset(
    records,
    training=True,
    batch_size=BATCH_SIZE
):
    """
    Build the TensorFlow dataset for garment attributes.
    """

    image_paths = [
        record[0]
        for record in records
    ]

    labels = [
        record[1]
        for record in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def load_attribute_image(image_path, label):

        image = preprocess_image(image_path)

        if training:
            image = augment_image(image)

        return image, label

    dataset = dataset.map(
        load_attribute_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=10000
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# BALANCE THE ATTRIBUTE TRAINING SET
# ============================================================

def balance_attribute_records(records, negatives_per_positive=1.0):
    """
    Keep every image that has at least one selected attribute, plus an
    equal number of randomly chosen images that have none, then shuffle
    them together.

    Why: with rare attributes, a random batch of 32 holds well under
    one image with a given attribute, so the model rarely sees what it is
    supposed to learn. After balancing, roughly half the images in every
    batch show at least one attribute.

    Only use this on the TRAINING split. The validation split stays as it
    is, so the evaluation reflects how common each attribute really is.
    """
    with_attribute = [r for r in records if sum(r[1]) > 0]
    without_attribute = [r for r in records if sum(r[1]) == 0]

    tf.random.set_seed(42)

    order = tf.random.shuffle(
        tf.range(len(without_attribute))
    ).numpy()

    keep = min(
        int(len(with_attribute) * negatives_per_positive),
        len(without_attribute)
    )

    chosen_negatives = [without_attribute[i] for i in order[:keep]]

    combined = with_attribute + chosen_negatives

    # Shuffle fully here: the dataset's shuffle buffer is smaller than the
    # list, so a list with all positives first would give early batches
    # that are almost all positives.
    mixed_order = tf.random.shuffle(
        tf.range(len(combined))
    ).numpy()

    print(
        f"Attribute training set balanced: "
        f"{len(with_attribute)} images with an attribute + "
        f"{len(chosen_negatives)} without."
    )

    return [combined[i] for i in mixed_order]


# ============================================================
# TRAINING / VALIDATION SPLIT
# ============================================================

def split_dataset(
    image_paths,
    labels,
    validation_ratio=0.2
):
    dataset = list(zip(image_paths, labels))

    tf.random.set_seed(42)

    indices = tf.random.shuffle(
        tf.range(len(dataset))
    ).numpy()

    shuffled_paths = [
        image_paths[i]
        for i in indices
    ]

    shuffled_labels = [
        labels[i]
        for i in indices
    ]

    total = len(shuffled_paths)

    split_index = int(
        total * (1 - validation_ratio)
    )

    train_paths = shuffled_paths[:split_index]
    train_labels = shuffled_labels[:split_index]

    val_paths = shuffled_paths[split_index:]
    val_labels = shuffled_labels[split_index:]

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels
    )

# ============================================================
# BUILD ALL TRAINING DATA
# ============================================================

def prepare_training_data():

    print("\nPreparing fabric condition dataset...")

    condition_paths, condition_labels = list_condition_images(
        CONDITION_DATASET
    )

    (
        condition_train_paths,
        condition_train_labels,
        condition_val_paths,
        condition_val_labels
    ) = split_dataset(
        condition_paths,
        condition_labels
    )

    condition_train = build_condition_dataset(
        condition_train_paths,
        condition_train_labels,
        training=True
    )

    condition_validation = build_condition_dataset(
        condition_val_paths,
        condition_val_labels,
        training=False
    )

    print(
        f"Condition training samples: "
        f"{len(condition_train_paths)}"
    )

    print(
        f"Condition validation samples: "
        f"{len(condition_val_paths)}"
    )


    print("\nPreparing DeepFashion attribute dataset...")

    attribute_records = build_attribute_records()

    attribute_paths = [
        record[0]
        for record in attribute_records
    ]

    attribute_labels = [
        record[1]
        for record in attribute_records
    ]

    (
        attribute_train_paths,
        attribute_train_labels,
        attribute_val_paths,
        attribute_val_labels
    ) = split_dataset(
        attribute_paths,
        attribute_labels
    )

    attribute_train_records = list(
        zip(
            attribute_train_paths,
            attribute_train_labels
        )
    )

    attribute_val_records = list(
        zip(
            attribute_val_paths,
            attribute_val_labels
        )
    )

    # Balance the TRAINING records only (validation stays natural)
    attribute_train_records = balance_attribute_records(
        attribute_train_records
    )

    attribute_train = build_attribute_dataset(
        attribute_train_records,
        training=True
    )

    attribute_validation = build_attribute_dataset(
        attribute_val_records,
        training=False
    )

    print(
        f"Attribute training samples: "
        f"{len(attribute_train_records)}"
    )

    print(
        f"Attribute validation samples: "
        f"{len(attribute_val_records)}"
    )


    return (
        condition_train,
        condition_validation,
        attribute_train,
        attribute_validation
    )


if __name__ == "__main__":

    print("\nTraining Data Pipeline")
    print("=" * 55)

    (
        condition_train,
        condition_validation,
        attribute_train,
        attribute_validation
    ) = prepare_training_data()


    print("\nChecking condition")

    condition_images, condition_labels = next(
        iter(condition_train)
    )

    print(
        "Condition images:",
        condition_images.shape
    )

    print(
        "Condition labels:",
        condition_labels.shape
    )


    print("\nChecking attribute")

    attribute_images, attribute_labels = next(
        iter(attribute_train)
    )

    print(
        "Attribute images:",
        attribute_images.shape
    )

    print(
        "Attribute labels:",
        attribute_labels.shape
    )


    print("\n" + "=" * 55)
    print("(Training) Image preprocessing completed.")