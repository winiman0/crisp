import tensorflow as tf

from src.model.backbone import build_backbone
from src.model.attributeH import attribute_head
from src.model.conditionH import build_condition_head


def build_model(input_shape=(224, 224, 3)):

    backbone = build_backbone(input_shape=input_shape)
    image_input = backbone.input

    # backbone now outputs TWO things — unpack both
    spatial_features, shared_features = backbone.output

    # Attribute head: still uses the pooled, global vector (holistic task)
    attribute_output = attribute_head(spatial_features)

    # Condition head: now uses the spatial map (localized task)
    condition_output = build_condition_head(spatial_features)

    model = tf.keras.Model(
        inputs=image_input,
        outputs={
            "attribute_output": attribute_output,
            "condition_output": condition_output
        },
        name="project_cnn"
    )

    return model


if __name__ == "__main__":
    print("\nCNN Model Test")
    print("=" * 55)

    model = build_model()
    model.summary()

    test_image = tf.random.uniform(shape=(1, 224, 224, 3), minval=0.0, maxval=1.0)
    outputs = model(test_image)

    print("\nOutput shapes:")
    print("Attribute output:", outputs["attribute_output"].shape)
    print("Condition output:", outputs["condition_output"].shape)