"""
Fair attribute evaluation: save as src/evaluation/evaluate_tuned.py

Why this exists
---------------
The normal attribute table cuts every attribute at 0.5. This model's highest
score on any attribute is about 0.3, so that table shows zeros even where the
model learned something.

This version:
  1. Splits the validation set into two halves.
  2. Picks the best threshold for each attribute on the FIRST half.
  3. Reports precision, recall and F1 on the SECOND half, which the threshold
     never saw, so the numbers are not flattered by the tuning.
  4. Adds AUC: the chance that a random image WITH the attribute gets a higher
     score than a random image WITHOUT it. 0.5 means no skill (a coin flip),
     1.0 means perfect ranking. It does not depend on any threshold.

Add to your evaluation file (eva.py):

    from src.evaluation.evaluate_tuned import evaluate_attributes_tuned
    ...
    evaluate_attributes_tuned(model, attribute_val_ds)
"""

import numpy as np
import tensorflow as tf

from src.dataset.attributemanage import SELECTED_ATTRIBUTES


# Includes low values on purpose: this model's scores are small, and the
# earlier grid (starting at 0.05) kept picking its lowest value.
THRESHOLD_GRID = [
    0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50
]


def precision_recall_f1(scores, labels, threshold):
    predicted = scores >= threshold

    tp = float(np.sum(predicted & (labels == 1)))
    fp = float(np.sum(predicted & (labels == 0)))
    fn = float(np.sum((~predicted) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def auc_score(scores, labels):
    """
    Chance that a random positive image scores higher than a random negative
    one, computed from the rank of each score (no library function).
    """
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))

    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    positive_rank_sum = float(np.sum(ranks[labels == 1]))

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def tuned_attribute_report(scores, labels, names):
    """
    scores, labels: arrays of shape (images, attributes)
    Returns the thresholds chosen on the first half.
    """
    total = scores.shape[0]
    half = total // 2

    tune_scores, test_scores = scores[:half], scores[half:]
    tune_labels, test_labels = labels[:half], labels[half:]

    print("\nAttribute Evaluation (tuned thresholds, held-out half)")
    print("=" * 84)
    print(
        f"Threshold chosen on {half} images, results measured on "
        f"{total - half} other images."
    )
    print("-" * 84)
    print(
        f"{'Attribute':<16}{'Images':>8}{'Threshold':>11}"
        f"{'Precision':>11}{'Recall':>9}{'F1':>8}{'AUC':>8}"
    )
    print("-" * 84)

    thresholds = []
    auc_values = []

    for i, name in enumerate(names):

        # 1. choose the threshold on the first half
        best_threshold = THRESHOLD_GRID[0]
        best_f1 = -1.0

        for threshold in THRESHOLD_GRID:
            _, _, f1 = precision_recall_f1(
                tune_scores[:, i], tune_labels[:, i], threshold
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)

        # 2. measure on the second half
        precision, recall, f1 = precision_recall_f1(
            test_scores[:, i], test_labels[:, i], best_threshold
        )
        auc = auc_score(test_scores[:, i], test_labels[:, i])
        auc_values.append(auc)

        positives = int(np.sum(test_labels[:, i] == 1))

        print(
            f"{name:<16}{positives:>8}{best_threshold:>11.3f}"
            f"{precision:>11.4f}{recall:>9.4f}{f1:>8.4f}{auc:>8.3f}"
        )

    print("-" * 84)
    valid_auc = [a for a in auc_values if a == a]  # drop NaN
    if valid_auc:
        print(f"Mean AUC: {sum(valid_auc) / len(valid_auc):.3f}   (0.5 = no skill)")
    print("'Images' = how many images in the measured half have the attribute.")
    print("Precision is understated: many real attributes are unlabelled in")
    print("DeepFashion, so correct detections can be counted as mistakes.")

    return thresholds


def evaluate_attributes_tuned(model, attribute_val_ds):

    all_scores = []
    all_labels = []

    for images, labels in attribute_val_ds:
        outputs = model(images, training=False)
        all_scores.append(outputs["attribute_output"].numpy())
        all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return tuned_attribute_report(scores, labels, SELECTED_ATTRIBUTES)