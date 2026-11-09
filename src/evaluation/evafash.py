"""
evaluate_fashionpedia.py   ->  save as  src/evaluation/evaluate_fashionpedia.py
Run:  python -m src.evaluation.evaluate_fashionpedia

Checks collar, pocket and zipper on the Fashionpedia VALIDATION images.
The model never trained on these images, and their part labels are cleaner
than DeepFashion's, so this is the fair test of whether adding Fashionpedia
helped. Compare the AUC here with the DeepFashion AUC (collar 0.564,
pocket 0.508, zipper 0.560 in the latest eva.py run).
"""

import numpy as np
import tensorflow as tf

from src.dataset.attributemanage import SELECTED_ATTRIBUTES
from src.dataset.fashionpedia import (
    build_fashionpedia_records,
    VAL_ANNOTATIONS,
    VAL_IMAGES,
    PART_TO_ATTRIBUTE,
)
from src.preprocessing.trainpre2 import preprocess_image, BATCH_SIZE
from src.evaluation.evaluate_tuned import (
    auc_score,
    precision_recall_f1,
    THRESHOLD_GRID,
)

MODEL_PATH = "project_model_best.keras"


def score_images(model, image_paths):
    """
    Runs every image through the model (no augmentation, no shuffling, so the
    scores line up with the list of paths) and returns the 26 attribute scores.
    """
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    all_scores = []

    for images in dataset:
        outputs = model(images, training=False)
        all_scores.append(outputs["attribute_output"].numpy())

    return np.concatenate(all_scores, axis=0)


def main():

    print("\nFashionpedia Validation: collar, pocket, zipper")
    print("=" * 76)

    records = build_fashionpedia_records(VAL_ANNOTATIONS, VAL_IMAGES)

    image_paths = [record[0] for record in records]
    labels = np.array([record[1] for record in records], dtype=np.float32)

    print("\nLoading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Scoring images...")
    scores = score_images(model, image_paths)

    # Shuffle once (fixed seed) before splitting in two halves, so the halves
    # are not ordered by file name. First half picks the threshold, second half
    # reports precision / recall / F1.
    order = np.random.RandomState(42).permutation(len(image_paths))
    half = len(order) // 2
    tune_rows, test_rows = order[:half], order[half:]

    print("\n" + "-" * 76)
    print(
        f"{'Attribute':<10}{'Images':>8}{'AUC':>8}{'Gap':>9}"
        f"{'Threshold':>11}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    )
    print("-" * 76)

    for name in PART_TO_ATTRIBUTE.values():

        slot = SELECTED_ATTRIBUTES.index(name)

        column_scores = scores[:, slot]
        column_labels = labels[:, slot]

        # Threshold-free check on all images
        auc = auc_score(column_scores, column_labels)

        # Average score on images that have the part minus those that do not
        gap = (
            column_scores[column_labels == 1].mean()
            - column_scores[column_labels == 0].mean()
        )

        # Pick the threshold on the first half
        best_threshold = THRESHOLD_GRID[0]
        best_f1 = -1.0

        for threshold in THRESHOLD_GRID:
            _, _, f1 = precision_recall_f1(
                column_scores[tune_rows], column_labels[tune_rows], threshold
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Report on the second half
        precision, recall, f1 = precision_recall_f1(
            column_scores[test_rows], column_labels[test_rows], best_threshold
        )

        images_with_part = int(np.sum(column_labels == 1))

        print(
            f"{name:<10}{images_with_part:>8}{auc:>8.3f}{gap:>9.4f}"
            f"{best_threshold:>11.3f}{precision:>11.4f}{recall:>9.4f}{f1:>8.4f}"
        )

    print("-" * 76)
    print("AUC and Gap use all validation images. Precision / Recall / F1 use")
    print("the held-out half. AUC 0.5 = no skill.")


if __name__ == "__main__":
    main()