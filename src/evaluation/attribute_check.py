"""
Attribute pipeline check. Run from the project root:

    python -m src.evaluation.attribute_check

(Save this file as src/evaluation/attribute_check.py.)

Two checks, in order:

  1. LABEL CHECK  - prints image paths that are labelled "button" so you can
                    open a few and confirm buttons are really there.
  2. OVERFIT TEST - trains the attribute head ONLY on 128 images (64 with a
                    button, 64 without) for many epochs. A working pipeline
                    must be able to memorize 128 images. If it cannot, the
                    problem is in the data pipeline or the model, and more
                    training will not fix it.
"""

import random
import tensorflow as tf

from src.preprocessing.trainpre import (
    build_attribute_records,
    build_attribute_dataset,
)
from src.model.model import build_model
from src.dataset.attributemanage import SELECTED_ATTRIBUTES


TEST_ATTRIBUTE = "button"
IMAGES_PER_GROUP = 64
BATCH = 16
EPOCHS = 40
LEARNING_RATE = 0.0003


def manual_binary_crossentropy(y_true, y_pred, epsilon=1e-7):
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -(
        y_true * tf.math.log(y_pred) +
        (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )
    return tf.reduce_mean(loss)


# ============================================================
# Check 1: do the labels match the pictures?
# ============================================================

def label_check(records, attribute_index, how_many=10):
    print("\nLABEL CHECK")
    print("=" * 70)
    print(f"First {how_many} images labelled '{TEST_ATTRIBUTE}'.")
    print("Open each file and confirm the garment really has that feature.\n")

    shown = 0
    for path, labels in records:
        if labels[attribute_index] == 1.0:
            positive_names = [
                SELECTED_ATTRIBUTES[i]
                for i, value in enumerate(labels)
                if value == 1.0
            ]
            print(path)
            print(f"    labelled: {', '.join(positive_names)}")
            shown += 1
            if shown >= how_many:
                break


# ============================================================
# Check 2: can the attribute head memorize a tiny set?
# ============================================================

def overfit_test(records, attribute_index):
    print("\nOVERFIT TEST")
    print("=" * 70)

    positives = [r for r in records if r[1][attribute_index] == 1.0]
    negatives = [r for r in records if r[1][attribute_index] == 0.0]

    random.seed(1)
    subset = (
        random.sample(positives, IMAGES_PER_GROUP) +
        random.sample(negatives, IMAGES_PER_GROUP)
    )
    random.shuffle(subset)

    print(f"Loading {len(subset)} images into memory...")
    dataset = build_attribute_dataset(subset, training=False, batch_size=BATCH)
    batches = list(dataset)
    print("Loaded.\n")

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(EPOCHS):
        losses = []

        for images, labels in batches:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)["attribute_output"]
                loss = manual_binary_crossentropy(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)

            # The condition head is unused here, so its gradients are None.
            pairs = [
                (g, v) for g, v in zip(gradients, model.trainable_variables)
                if g is not None
            ]
            optimizer.apply_gradients(pairs)
            losses.append(float(loss))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:>2}/{EPOCHS}  loss: {sum(losses) / len(losses):.4f}")

    # --------------------------------------------------------
    # Score the SAME 128 training images two ways:
    #   training=True  -> BatchNorm uses the statistics of the current batch
    #                     (this is how the model was trained)
    #   training=False -> BatchNorm uses its stored running averages
    #                     (this is how evaluate.py and the app run the model)
    # If these two disagree, BatchNorm's stored statistics are the problem.
    # --------------------------------------------------------
    def average_gap(training_mode):
        pos_sum, pos_n, neg_sum, neg_n = 0.0, 0.0, 0.0, 0.0

        for images, labels in batches:
            scores = model(images, training=training_mode)["attribute_output"][:, attribute_index]
            actual = labels[:, attribute_index]

            pos_sum += float(tf.reduce_sum(scores * actual))
            pos_n += float(tf.reduce_sum(actual))
            neg_sum += float(tf.reduce_sum(scores * (1.0 - actual)))
            neg_n += float(tf.reduce_sum(1.0 - actual))

        pos_mean = pos_sum / max(pos_n, 1.0)
        neg_mean = neg_sum / max(neg_n, 1.0)
        return pos_mean, neg_mean, pos_mean - neg_mean

    print(f"\nScores for '{TEST_ATTRIBUTE}' on the 128 training images:")
    print(f"{'Mode':<34}{'Has it':>10}{'Does not':>12}{'Gap':>10}")

    train_pos, train_neg, train_gap = average_gap(True)
    print(f"{'training=True  (batch statistics)':<34}{train_pos:>10.4f}{train_neg:>12.4f}{train_gap:>10.4f}")

    infer_pos, infer_neg, infer_gap = average_gap(False)
    print(f"{'training=False (stored statistics)':<34}{infer_pos:>10.4f}{infer_neg:>12.4f}{infer_gap:>10.4f}")

    # Show the BatchNorm settings in the model
    # (the backbone may be a model nested inside the main model, so walk
    # through nested layers too)
    def find_batchnorm_layers(parent):
        found = []
        for layer in parent.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                found.append(layer)
            elif hasattr(layer, "layers"):
                found.extend(find_batchnorm_layers(layer))
        return found

    bn_layers = find_batchnorm_layers(model)
    if bn_layers:
        print(f"\nBatchNorm layers in model: {len(bn_layers)}, momentum = {bn_layers[0].momentum}")

    print()
    if train_gap > 0.5 and infer_gap < 0.2:
        print("RESULT: BatchNorm problem. The model memorizes fine with batch")
        print("statistics but fails with its stored statistics. The pipeline and")
        print("labels are OK; the stored statistics are wrong.")
    elif train_gap > 0.5 and infer_gap > 0.5:
        print("RESULT: the model can memorize in both modes. No bug found;")
        print("the main-run problem is training volume / signal strength.")
    else:
        print("RESULT: the model cannot separate the images even with batch")
        print("statistics. Suspect the labels or the model; open the label-check")
        print("images and look for wrong labels.")


if __name__ == "__main__":
    all_records = build_attribute_records()
    index = SELECTED_ATTRIBUTES.index(TEST_ATTRIBUTE)

    label_check(all_records, index)
    overfit_test(all_records, index)