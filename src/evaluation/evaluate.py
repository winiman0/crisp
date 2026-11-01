import tensorflow as tf

from src.preprocessing.trainpre import prepare_training_data
from src.model.model import build_model


MODEL_PATH = "project_model_best.keras"


# ============================================================
# Condition evaluation
# ============================================================

def evaluate_condition(
    model,
    condition_val_ds
):
    correct = 0
    total = 0

    confusion_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    for images, labels in condition_val_ds:

        outputs = model(
            images,
            training=False
        )

        predictions = outputs["condition_output"]

        predicted_classes = tf.argmax(
            predictions,
            axis=1
        )

        for actual, predicted in zip(
            labels.numpy(),
            predicted_classes.numpy()
        ):
            actual = int(actual)
            predicted = int(predicted)

            confusion_matrix[actual][predicted] += 1

            if actual == predicted:
                correct += 1

            total += 1

    accuracy = correct / total

    print("\nCondition Evaluation")
    print("=" * 55)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix")
    print("                Predicted")
    print("              Clean Stain Defect")
    print(
        f"Actual Clean   "
        f"{confusion_matrix[0][0]:5d} "
        f"{confusion_matrix[0][1]:5d} "
        f"{confusion_matrix[0][2]:5d}"
    )
    print(
        f"Actual Stain   "
        f"{confusion_matrix[1][0]:5d} "
        f"{confusion_matrix[1][1]:5d} "
        f"{confusion_matrix[1][2]:5d}"
    )
    print(
        f"Actual Defect  "
        f"{confusion_matrix[2][0]:5d} "
        f"{confusion_matrix[2][1]:5d} "
        f"{confusion_matrix[2][2]:5d}"
    )

    # --------------------------------------------------------
    # Precision, Recall and F1-score
    # --------------------------------------------------------

    class_names = [
        "Clean",
        "Stain",
        "Defect"
    ]

    print("\nClass Metrics")
    print("-" * 55)

    for class_index, class_name in enumerate(class_names):

        true_positive = confusion_matrix[
            class_index
        ][class_index]

        false_positive = 0
        false_negative = 0

        for i in range(3):
            if i != class_index:
                false_positive += confusion_matrix[i][class_index]
                false_negative += confusion_matrix[class_index][i]

        if true_positive + false_positive > 0:
            precision = (
                true_positive /
                (true_positive + false_positive)
            )
        else:
            precision = 0.0

        if true_positive + false_negative > 0:
            recall = (
                true_positive /
                (true_positive + false_negative)
            )
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = (
                2 * precision * recall /
                (precision + recall)
            )
        else:
            f1 = 0.0

        print(f"\n{class_name}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")


# ============================================================
# Attribute evaluation
# ============================================================

def evaluate_attributes(
    model,
    attribute_val_ds
):
    attribute_correct = None
    attribute_total = 0

    for images, labels in attribute_val_ds:

        outputs = model(
            images,
            training=False
        )

        predictions = outputs["attribute_output"]

        predicted_labels = tf.cast(
            predictions >= 0.5,
            tf.float32
        )

        correct = tf.equal(
            predicted_labels,
            labels
        )

        correct = tf.reduce_sum(
            tf.cast(correct, tf.int32),
            axis=0
        )

        if attribute_correct is None:
            attribute_correct = correct
        else:
            attribute_correct += correct

        attribute_total += labels.shape[0]

    attribute_accuracies = (
        attribute_correct.numpy() /
        attribute_total
    )

    print("\nAttribute Evaluation")
    print("=" * 55)

    for index, accuracy in enumerate(
        attribute_accuracies
    ):
        print(
            f"Attribute {index + 1:2d}: "
            f"{accuracy:.4f}"
        )

    overall_accuracy = (
        attribute_correct.numpy().sum() /
        (
            attribute_total *
            len(attribute_accuracies)
        )
    )

    print(
        f"\nOverall Attribute Accuracy: "
        f"{overall_accuracy:.4f}"
    )

"""
Two new functions to paste into src/evaluation/evaluate.py.

They use only what evaluate.py already imports (tensorflow, SELECTED_ATTRIBUTES).
Call both from your main evaluation function after the model is loaded, e.g.:

    check_attribute_separation(model, attribute_val_ds)
    find_best_thresholds(model, attribute_val_ds)
"""


# ============================================================
# Check 1: did the model learn anything beyond "how common is it"?
# ============================================================

def check_attribute_separation(model, attribute_val_ds):
    """
    For each attribute, compute:
        average score on images that HAVE the attribute
        average score on images that DO NOT have it
    If the two averages are almost the same, the model cannot tell them
    apart, no matter which threshold is used.
    """
    n = len(SELECTED_ATTRIBUTES)

    positive_sum = tf.zeros(n)
    positive_count = tf.zeros(n)
    negative_sum = tf.zeros(n)
    negative_count = tf.zeros(n)

    for images, labels in attribute_val_ds:
        scores = model(images, training=False)["attribute_output"]

        positive_sum += tf.reduce_sum(scores * labels, axis=0)
        positive_count += tf.reduce_sum(labels, axis=0)
        negative_sum += tf.reduce_sum(scores * (1.0 - labels), axis=0)
        negative_count += tf.reduce_sum(1.0 - labels, axis=0)

    positive_mean = positive_sum / tf.maximum(positive_count, 1.0)
    negative_mean = negative_sum / tf.maximum(negative_count, 1.0)
    gap = positive_mean - negative_mean

    print("\nAttribute Separation (average score: has attribute vs does not)")
    print("=" * 70)
    print(f"{'Attribute':<16}{'Has it':>10}{'Does not':>12}{'Gap':>10}   Verdict")
    print("-" * 70)

    for i, name in enumerate(SELECTED_ATTRIBUTES):
        g = float(gap[i])
        # Rule of thumb only: a gap under 0.02 means the model barely
        # distinguishes the two groups.
        verdict = "learned something" if g >= 0.02 else "NOT learned"
        print(f"{name:<16}{float(positive_mean[i]):>10.4f}{float(negative_mean[i]):>12.4f}{g:>10.4f}   {verdict}")


# ============================================================
# Check 2: pick a threshold per attribute instead of always 0.5
# ============================================================

def find_best_thresholds(model, attribute_val_ds):
    """
    For every attribute, try thresholds 0.05, 0.10, ... 0.95 and keep the
    one with the best F1-score. Rare attributes rarely reach 0.5, so a
    fixed 0.5 line unfairly marks them as "never present".

    Note: the thresholds are chosen on the same validation set they are
    reported on, so the F1 shown here is a little optimistic. For the
    final report, choose thresholds on one half of the validation set and
    report results on the other half.
    """
    n = len(SELECTED_ATTRIBUTES)

    all_scores = []
    all_labels = []
    for images, labels in attribute_val_ds:
        scores = model(images, training=False)["attribute_output"]
        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = tf.concat(all_scores, axis=0)
    all_labels = tf.concat(all_labels, axis=0)

    candidate_thresholds = [round(0.05 * k, 2) for k in range(1, 20)]
    best_thresholds = []

    print("\nBest Threshold Per Attribute")
    print("=" * 70)
    print(f"{'Attribute':<16}{'Threshold':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("-" * 70)

    for i, name in enumerate(SELECTED_ATTRIBUTES):
        scores_i = all_scores[:, i]
        labels_i = all_labels[:, i]

        best_f1 = -1.0
        best = (0.5, 0.0, 0.0, 0.0)

        for threshold in candidate_thresholds:
            predicted = tf.cast(scores_i >= threshold, tf.float32)

            tp = float(tf.reduce_sum(predicted * labels_i))
            fp = float(tf.reduce_sum(predicted * (1.0 - labels_i)))
            fn = float(tf.reduce_sum((1.0 - predicted) * labels_i))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best = (threshold, precision, recall, f1)

        best_thresholds.append(best[0])
        print(f"{name:<16}{best[0]:>10.2f}{best[1]:>12.4f}{best[2]:>10.4f}{best[3]:>10.4f}")

    return best_thresholds

# ============================================================
# Main evaluation
# ============================================================

def evaluate_model():

    print("\nModel Evaluation")
    print("=" * 55)

    print("\nLoading validation datasets...")

    (
        condition_train_ds,
        condition_val_ds,
        attribute_train_ds,
        attribute_val_ds
    ) = prepare_training_data()

    print("Validation datasets loaded.")

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    evaluate_condition(
        model,
        condition_val_ds
    )

    evaluate_attributes(
        model,
        attribute_val_ds
    )

    check_attribute_separation(model, attribute_val_ds)
    find_best_thresholds(model, attribute_val_ds)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    evaluate_model()