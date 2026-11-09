"""
Step 5: Data pipeline — loading, preprocessing, and augmentation.

This matches your Preliminary Study + Design phase decisions:
    - resize to 224x224x3
    - pixel normalization
    - augmentation to simulate real-world fabric conditions (folds, lighting)

Expected folder structure (adjust to match your actual dataset):

    dataset/
        train/
            images/            <- raw clothing images
            labels.csv         <- columns: filename, category, condition
        val/
            images/
            labels.csv
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os

IMG_SIZE = (224, 224)


def load_labels(csv_path):
    """
    Reads the labels CSV. Expects columns: filename, category, condition
    e.g.  shirt_001.jpg, shirt, stain
    """
    df = pd.read_csv(csv_path)
    return df


def preprocess_image(image_path):
    """
    Loads a single image from disk, resizes it, and normalizes pixel values
    to the [0, 1] range (equivalent to what OpenCV normalization does in
    your proposal's Image Processing Module).
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # normalize pixel values to [0, 1]
    return image


def augment_image(image):
    """
    Basic augmentation to simulate real-world fabric conditions:
    lighting shifts, minor rotations, and flips — this is the
    preprocessing step your literature review implication called for,
    to reduce the model confusing folds/lighting with real defects.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)   # lighting variation
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    return image


def build_dataset(csv_path, images_dir, category_to_idx, condition_to_idx,
                   batch_size=32, training=True):
    """
    Builds a tf.data.Dataset yielding:
        (image, {"garment_category_output": category_label,
                 "fabric_condition_output": condition_label})

    which matches the two named outputs from build_garment_cnn().
    """
    df = load_labels(csv_path)

    filepaths = [os.path.join(images_dir, f) for f in df["filename"]]
    category_labels = df["category"].map(category_to_idx).values
    condition_labels = df["condition"].map(condition_to_idx).values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, category_labels, condition_labels))

    def _load(fp, cat, cond):
        image = preprocess_image(fp)
        if training:
            image = augment_image(image)
        return image, {"garment_category_output": cat, "fabric_condition_output": cond}

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(df))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Example label mappings — replace with your actual dataset's categories
CATEGORY_TO_IDX = {"shirt": 0, "trousers": 1, "jacket": 2, "dress": 3, "skirt": 4}
CONDITION_TO_IDX = {"clean": 0, "stain": 1, "defect": 2}  # matches Table 3.2