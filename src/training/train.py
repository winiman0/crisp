import tensorflow as tf

from src.preprocessing.trainpre2 import prepare_training_data
from src.model.model import build_model
from src.dataset.attributemanage import SELECTED_ATTRIBUTES


# ============================================================
# Training settings
# ============================================================

EPOCHS = 20
LEARNING_RATE = 0.0001
STEPS_PER_EPOCH = 500
VALIDATION_STEPS = 20
NUM_ATTRIBUTES = len(SELECTED_ATTRIBUTES)

# --- Loss balance between the two tasks ---------------------------------
# The attribute loss is now ~0.75 (it used to be tiny), so it can drown out
# the condition task in the shared backbone. Scaling it down keeps the
# condition task in charge, like in your earlier good run.
ATTRIBUTE_LOSS_WEIGHT = 1.0
CONDITION_LOSS_WEIGHT = 3.0

# Cap on the per-attribute positive weight. The attribute training set is now
# balanced in trainpre.py (about half the images have an attribute), so the
# rebalancing is already done there. A small cap avoids correcting twice.
ATTRIBUTE_POS_WEIGHT_CAP = 2.0

# --- Stability settings --------------------------------------------------
# Gradient clipping: if one batch produces a huge update (rare attributes
# with big weights can do this), shrink it so it cannot knock the weights
# off course. Written by hand below.
MAX_GRADIENT_NORM = 5.0

# Learning-rate decay: when the best score has not improved for this many
# epochs in a row, halve the learning rate.
#
# Both the decay and early stopping are driven by the CONDITION score. In
# your last run that score stalled, so the learning rate was cut and training
# stopped at epoch 10 (1,500 of the planned 3,000 steps), which starved the
# attribute head. For this test run both are set above EPOCHS, which switches
# them off. Set LR_DECAY_PATIENCE = 2 and EARLY_STOP_PATIENCE = 6 to turn them
# back on later.
LR_DECAY_PATIENCE = 100
EARLY_STOP_PATIENCE = 100
LR_DECAY_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# A checkpoint only counts as "best" if Stain recall is at least this high.
# Stain is the hardest class and the core of the project, so a model that
# wins on macro F1 by ignoring stains should not be saved.
MIN_STAIN_RECALL = 0.70


# ============================================================
# Loss functions
# ============================================================

condition_loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
    reduction="none"
)
CONDITION_CLASS_WEIGHTS = tf.constant(
    [0.87, 1.17, 1.01],
    dtype=tf.float32
)


def weighted_binary_crossentropy(y_true, y_pred, pos_weight, mask=None, epsilon=1e-7):
    """
    Manual weighted binary cross-entropy (no built-in pos_weight loss class,
    to stay consistent with the no-built-in-tools rule), with a mask.

    Standard weighted BCE formula:
        loss = -[pos_weight * y * log(p) + (1 - y) * log(1 - p)]

    pos_weight is a length-26 vector, one value per attribute. For a rare
    attribute, pos_weight is large, so the model is penalized heavily for
    missing the few real positives; this is what stops it from collapsing to
    "always predict absent."

    mask has the same shape as the labels. 1 = this label is known for this
    image, 0 = unknown, so it adds nothing to the loss. Fashionpedia images
    only know collar, pocket and zipper; DeepFashion images know all 26.
    The loss is averaged over the KNOWN labels only. With no mask (validation)
    every label counts, which gives the same value as before.
    """
    if mask is None:
        mask = tf.ones_like(y_true)

    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -(
        pos_weight * y_true * tf.math.log(y_pred) +
        (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )

    loss = loss * mask

    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def compute_attribute_pos_weights(positives, labelled, cap=20.0):
    """
    Turns the label counts (made in trainpre2.py) into one weight per
    attribute:

        pos_weight = negative_count / positive_count

    where negative_count = labelled - positives. "labelled" only counts the
    images that actually give an answer for that attribute, so Fashionpedia
    images do not count as negatives for attributes they say nothing about.

    The weight is capped at `cap` so a handful of rare examples cannot
    dominate every gradient step.
    """
    positives = tf.maximum(tf.constant(positives, dtype=tf.float32), 1.0)
    labelled = tf.constant(labelled, dtype=tf.float32)

    negatives = labelled - positives

    pos_weight = negatives / positives
    pos_weight = tf.clip_by_value(pos_weight, 1.0, cap)

    print("\nAttribute positive-class weights (capped at {:.0f}):".format(cap))
    for name, w in zip(SELECTED_ATTRIBUTES, pos_weight.numpy()):
        print(f"  {name:<15} weight: {w:.2f}")

    return pos_weight


# ============================================================
# Train one step using both tasks
# ============================================================

def train_step(
    model,
    attribute_model,
    condition_model,
    optimizer,
    attribute_images,
    attribute_labels,
    attribute_mask,
    condition_images,
    condition_labels,
    attribute_pos_weight
):
    # attribute_model and condition_model are two views of the SAME model.
    # They share the backbone and its weights. Each one stops at its own
    # head, so a garment photo never passes through the condition head's
    # batch-norm layers (and a fabric image never passes through the
    # attribute head). Before, every garment photo went through the whole
    # model and silently changed the condition head's stored statistics.

    with tf.GradientTape() as tape:

        attribute_predictions = attribute_model(attribute_images, training=True)
        attribute_loss = weighted_binary_crossentropy(
            attribute_labels, attribute_predictions, attribute_pos_weight,
            mask=attribute_mask
        )

        condition_predictions = condition_model(condition_images, training=True)

        condition_losses = condition_loss_function(condition_labels, condition_predictions)
        sample_weights = tf.gather(CONDITION_CLASS_WEIGHTS, condition_labels)
        condition_loss = tf.reduce_mean(condition_losses * sample_weights)

        total_loss = (
            ATTRIBUTE_LOSS_WEIGHT * attribute_loss +
            CONDITION_LOSS_WEIGHT * condition_loss
        )

    gradients = tape.gradient(total_loss, model.trainable_variables)

    # Manual gradient clipping by global norm:
    #   measure the overall size of the update; if it is bigger than the
    #   allowed maximum, scale every gradient down by the same ratio.
    global_norm = tf.linalg.global_norm(gradients)
    scale = tf.minimum(1.0, MAX_GRADIENT_NORM / (global_norm + 1e-6))
    gradients = [g * scale for g in gradients]

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return attribute_loss, condition_loss, total_loss


# ============================================================
# Validation
# ============================================================

def validate_model(model, attribute_val_ds, condition_val_ds, attribute_pos_weight):

    attribute_losses = []
    condition_losses = []

    for images, labels in attribute_val_ds.take(VALIDATION_STEPS):
        outputs = model(images, training=False)
        loss = weighted_binary_crossentropy(labels, outputs["attribute_output"], attribute_pos_weight)
        attribute_losses.append(loss)

    for images, labels in condition_val_ds.take(VALIDATION_STEPS):
        outputs = model(images, training=False)
        batch_condition_losses = condition_loss_function(labels, outputs["condition_output"])
        sample_weights = tf.gather(CONDITION_CLASS_WEIGHTS, labels)
        loss = tf.reduce_mean(batch_condition_losses * sample_weights)
        condition_losses.append(loss)

    attribute_val_loss = tf.reduce_mean(attribute_losses)
    condition_val_loss = tf.reduce_mean(condition_losses)
    total_val_loss = (
        ATTRIBUTE_LOSS_WEIGHT * attribute_val_loss +
        CONDITION_LOSS_WEIGHT * condition_val_loss
    )

    return attribute_val_loss, condition_val_loss, total_val_loss


def calculate_condition_metrics(model, condition_val_ds):
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for images, labels in condition_val_ds:
        outputs = model(images, training=False)
        predictions = tf.argmax(outputs["condition_output"], axis=1)

        for actual, predicted in zip(labels.numpy(), predictions.numpy()):
            confusion_matrix[int(actual)][int(predicted)] += 1

    f1_scores = []
    for class_index in range(3):
        tp = confusion_matrix[class_index][class_index]
        fp = sum(confusion_matrix[i][class_index] for i in range(3) if i != class_index)
        fn = sum(confusion_matrix[class_index][i] for i in range(3) if i != class_index)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / 3

    tp = confusion_matrix[1][1]
    fn = confusion_matrix[1][0] + confusion_matrix[1][2]
    stain_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return macro_f1, stain_recall


# ============================================================
# Main training function
# ============================================================

def train_model():

    print("\nModel Training")
    print("=" * 55)

    print("\nLoading training datasets...")
    (
        condition_train_ds,
        condition_val_ds,
        attribute_train_ds,
        attribute_val_ds,
        attribute_counts
    ) = prepare_training_data(for_training=True)
    print("Training datasets loaded.")

    # --------------------------------------------------------
    # Attribute positive-class weights, from the label counts that
    # trainpre2.py made (no need to loop over every image any more).
    # --------------------------------------------------------
    print("\nComputing attribute class weights...")
    positives, labelled = attribute_counts
    attribute_pos_weight = compute_attribute_pos_weights(
        positives, labelled, cap=ATTRIBUTE_POS_WEIGHT_CAP
    )

    print("\nBuilding CNN model...")
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Two views of the same model (same layers, same weights), one per task
    # (model.output is a dictionary; the keys are the same ones evaluate.py
    # uses, so they are known to exist.)
    attribute_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.output["attribute_output"]
    )
    condition_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.output["condition_output"]
    )
    print("CNN model created.")

    attribute_train_ds = attribute_train_ds.repeat()
    condition_train_ds = condition_train_ds.repeat()

    attribute_iterator = iter(attribute_train_ds)
    condition_iterator = iter(condition_train_ds)
    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    has_saved_once = False
    PATIENCE = EARLY_STOP_PATIENCE

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 55)

        attribute_losses = []
        condition_losses = []
        total_losses = []

        for step in range(STEPS_PER_EPOCH):
            attribute_images, attribute_labels, attribute_mask = next(attribute_iterator)
            condition_images, condition_labels = next(condition_iterator)

            attribute_loss, condition_loss, total_loss = train_step(
                model, attribute_model, condition_model, optimizer,
                attribute_images, attribute_labels, attribute_mask,
                condition_images, condition_labels,
                attribute_pos_weight
            )

            attribute_losses.append(attribute_loss)
            condition_losses.append(condition_loss)
            total_losses.append(total_loss)

        attribute_train_loss = tf.reduce_mean(attribute_losses)
        condition_train_loss = tf.reduce_mean(condition_losses)
        total_train_loss = tf.reduce_mean(total_losses)

        attribute_val_loss, condition_val_loss, total_val_loss = validate_model(
            model, attribute_val_ds, condition_val_ds, attribute_pos_weight
        )

        print(f"Attribute Loss: {attribute_train_loss:.4f} | Validation: {attribute_val_loss:.4f}")
        print(f"Condition Loss: {condition_train_loss:.4f} | Validation: {condition_val_loss:.4f}")
        print(f"Total Training Loss: {total_train_loss:.4f} | Total Validation Loss: {total_val_loss:.4f}")

        macro_f1, stain_recall = calculate_condition_metrics(model, condition_val_ds)
        print(f"Condition Macro F1: {macro_f1:.4f}")
        print(f"Stain Recall: {stain_recall:.4f}")

        # An epoch only qualifies as "best" if Stain recall is acceptable AND
        # macro F1 beats the previous best.
        qualifies = (stain_recall >= MIN_STAIN_RECALL) and (macro_f1 > best_macro_f1)

        if qualifies:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            has_saved_once = True
            model.save("project_model_best.keras")
            print(f"  -> New best Macro F1 ({best_macro_f1:.4f}) with Stain Recall {stain_recall:.4f}. Model saved.")
        elif not has_saved_once:
            # Early epochs: the model has not reached the Stain recall floor
            # yet. This is normal while it is still learning, so it must not
            # count as "stalled" (otherwise the learning rate is cut and
            # early stopping starts before training has really begun).
            print(f"  -> Not saved yet: Stain Recall {stain_recall:.4f} is below {MIN_STAIN_RECALL:.2f}. Still waiting for the first qualifying epoch.")
        else:
            epochs_without_improvement += 1
            if stain_recall < MIN_STAIN_RECALL:
                print(f"  -> Not saved: Stain Recall {stain_recall:.4f} is below {MIN_STAIN_RECALL:.2f}.")
            else:
                print(f"  -> No Macro F1 improvement for {epochs_without_improvement} epoch(s).")

            # Halve the learning rate every LR_DECAY_PATIENCE stalled epochs
            if epochs_without_improvement % LR_DECAY_PATIENCE == 0:
                current_lr = float(optimizer.learning_rate.numpy())
                new_lr = max(current_lr * LR_DECAY_FACTOR, MIN_LEARNING_RATE)
                optimizer.learning_rate.assign(new_lr)
                print(f"  -> Learning rate lowered: {current_lr:.2e} -> {new_lr:.2e}")

            if epochs_without_improvement >= PATIENCE:
                print(f"\nStopping early: no improvement for {PATIENCE} epochs.")
                break

    print("\nTraining completed.")
    print(f"Best model saved as: project_model_best.keras (Macro F1 {best_macro_f1:.4f})")
    return model


if __name__ == "__main__":
    train_model()