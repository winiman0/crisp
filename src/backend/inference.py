import os
import sys
import numpy as np
import tensorflow as tf

from src.preprocessing.trainpre import preprocess_image
from src.dataset.attributemanage import SELECTED_ATTRIBUTES


MODEL_PATH = "project_model.keras"

CONDITION_CLASSES = [
    "clean",
    "stain",
    "defect"
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


def predict_image(model, image_path):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file was not found:\n{image_path}"
        )

    image = preprocess_image(image_path)

    image = tf.expand_dims(
        image,
        axis=0
    )

    outputs = model(
        image,
        training=False
    )

    attribute_predictions = (
        outputs["attribute_output"][0].numpy()
    )

    condition_predictions = (
        outputs["condition_output"][0].numpy()
    )

    condition_index = int(
        np.argmax(condition_predictions)
    )

    condition_name = CONDITION_CLASSES[
        condition_index
    ]

    return (
        attribute_predictions,
        condition_predictions,
        condition_name
    )


def display_results(
    image_path,
    attribute_predictions,
    condition_predictions,
    condition_name
):

    print("\n==========================================")
    print("Project Aura - Image Prediction")
    print("==========================================")

    print(f"\nImage:")
    print(image_path)

    print("\nCondition Prediction")
    print("------------------------------------------")

    for class_name, probability in zip(
        CONDITION_CLASSES,
        condition_predictions
    ):
        print(
            f"{class_name:<10}: "
            f"{probability * 100:.2f}%"
        )

    print(
        f"\nPredicted condition: "
        f"{condition_name.upper()}"
    )

    print("\nAttribute Predictions")
    print("------------------------------------------")

    detected_count = 0

    for attribute, probability in zip(
        SELECTED_ATTRIBUTES,
        attribute_predictions
    ):

        if probability >= 0.5:
            print(
                f"{attribute:<20}: "
                f"{probability * 100:.2f}%"
            )
            detected_count += 1

    if detected_count == 0:
        print("No attributes passed the 0.50 threshold.")

    print("\n==========================================")


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:"
        )
        print(
            "python -m src.inference.predict "
            "\"path_to_image\""
        )
        return

    image_path = sys.argv[1]

    print("\nLoading model...")

    model = load_model()

    print("Model loaded successfully.")

    (
        attribute_predictions,
        condition_predictions,
        condition_name
    ) = predict_image(
        model,
        image_path
    )

    display_results(
        image_path,
        attribute_predictions,
        condition_predictions,
        condition_name
    )


if __name__ == "__main__":
    main()