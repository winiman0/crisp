"""
1  = attribute is present
-1  = attribute is absent
0  = attribute is unknown

"""

import os

# Paths

DEEPFASHION_DIR = (
    r"C:\Users\User\crisp\dataset"
)

ANNOTATION_DIR = os.path.join(
    DEEPFASHION_DIR,
    "Category and Attribute Prediction Benchmark\Anno_coarse"
)

ATTRIBUTE_DEFINITION_FILE = os.path.join(
    ANNOTATION_DIR,
    "list_attr_cloth.txt"
)

IMAGE_ATTRIBUTE_FILE = os.path.join(
    ANNOTATION_DIR,
    "list_attr_img.txt"
)

IMAGE_ROOT = DEEPFASHION_DIR

#Attributes selected 

SELECTED_ATTRIBUTES = [

    # Garment structure
    "button",
    "zipper",
    "collar",
    "pocket",

    # Colour
    "red",
    "pink",
    "dark",
    "light",

    #Pattern
    "floral",
    "striped",
    "plaid",
    "polka dot",

    # Sleeve
    "long sleeve",
    "long-sleeved",
    "sleeveless",

    # Neckline
    "crew neck",
    "v-neck",
    "boat neck",
    "scoop-neck",
    "turtle-neck",

    #Fabric
    "cotton",
    "denim",
    "chiffon",
    "leather",
    "knit",
    "lace",
]

#Annotation values
VALID_LABELS = {
    "-1",
    "0",
    "1",
}

# Definitions
def read_attribute_definitions(file_path):
    
    #Reads list_attr_cloth.txt
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Attribute definition file was not found:\n"
            f"{file_path}"
        )

    attributes = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        #number of attributes
        first_line = file.readline().strip()

        if not first_line:
            raise ValueError(
                "The attribute definition file is empty."
            )

        expected_attributes = int(first_line)

        #column heading
        file.readline()

        for line_number, line in enumerate(
            file,
            start=3
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.rsplit(
                maxsplit=1
            )

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid attribute definition at "
                    f"line {line_number}:\n{line}"
                )

            attribute_name = parts[0].strip()

            attributes.append(
                attribute_name
            )

    if len(attributes) != expected_attributes:
        raise ValueError(
            f"Attribute count mismatch.\n"
            f"Expected: {expected_attributes}\n"
            f"Found: {len(attributes)}"
        )

    return attributes


# Find selected attribute
def find_selected_attributes(all_attributes,selected_attributes):
  
    #Finds the zero-based position of every selected attribute.
    attribute_positions = {}

    for attribute in selected_attributes:

        if attribute not in all_attributes:
            raise ValueError(
                f"Selected attribute '{attribute}' "
                f"was not found in list_attr_cloth.txt."
            )

        position = all_attributes.index(
            attribute
        )

        attribute_positions[
            attribute
        ] = position

    return attribute_positions



# Display selected attributes n positions
def display_selected_attributes(attribute_positions):
    print("\nSelected attributes:")
    print("----------------------------------")

    for attribute, position in (
        attribute_positions.items()
    ):

        print(
            f"  {attribute:<20}"
            f" column {position + 1}"
        )


# Validate the annotations
def validate_image_annotations(
    image_attribute_file,
    all_attributes,
    attribute_positions,
    image_root
):

    if not os.path.exists(image_attribute_file):
        raise FileNotFoundError(
            f"Image attribute file was not found:\n"
            f"{image_attribute_file}"
        )

    with open(
        image_attribute_file,
        "r",
        encoding="utf-8"
    ) as file:

        #expected number of images
        first_line = file.readline().strip()

        if not first_line:
            raise ValueError(
                "The image attribute file is empty."
            )

        expected_images = int(first_line)

        #column heading
        file.readline()

        processed_images = 0
        invalid_label_count = 0
        missing_image_count = 0

        # Count selected labels
        attribute_statistics = {}

        for attribute in attribute_positions:
            attribute_statistics[attribute] = {
                "present": 0,
                "absent": 0,
                "unknown": 0,
            }

        #Process image
        for line_number, line in enumerate(
            file,
            start=3
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # separate image path and labels
            image_path = parts[0]
            labels = parts[1:]

            # Check number of labels
            if len(labels) != len(all_attributes):

                raise ValueError(
                    f"Invalid annotation at line "
                    f"{line_number}.\n"
                    f"Image: {image_path}\n"
                    f"Expected {len(all_attributes)} labels, "
                    f"found {len(labels)}."
                )

            # Check label values

            for label in labels:

                if label not in VALID_LABELS:

                    invalid_label_count += 1

                    raise ValueError(
                        f"Invalid label '{label}' "
                        f"at line {line_number}.\n"
                        f"Image: {image_path}\n"
                        f"Valid labels are: "
                        f"{sorted(VALID_LABELS)}"
                    )

            # Check if image exists
            full_image_path = os.path.join(
                image_root,
                image_path
            )

            if not os.path.isfile(
                full_image_path
            ):

                missing_image_count += 1

                raise FileNotFoundError(
                    f"Image referenced in annotation "
                    f"file was not found:\n"
                    f"{full_image_path}"
                )

            # Count selected attributes
            for attribute, position in (
                attribute_positions.items()
            ):

                label = labels[position]

                if label == "1":
                    attribute_statistics[
                        attribute
                    ]["present"] += 1

                elif label == "-1":
                    attribute_statistics[
                        attribute
                    ]["absent"] += 1

                elif label == "0":
                    attribute_statistics[
                        attribute
                    ]["unknown"] += 1

            processed_images += 1


    # Final annotation checks
    if processed_images != expected_images:
        raise ValueError(
            f"Image count mismatch.\n"
            f"Expected: {expected_images}\n"
            f"Processed: {processed_images}"
        )

    if invalid_label_count != 0:
        raise ValueError(
            f"Invalid labels found: "
            f"{invalid_label_count}"
        )

    if missing_image_count != 0:
        raise FileNotFoundError(
            f"Missing images found: "
            f"{missing_image_count}"
        )

    return (
        expected_images,
        attribute_statistics
    )


# Display
def display_attribute_statistics(
    attribute_statistics,
    total_images
):

    print("\nSelected attribute label :")
    print("--------------------------------------")

    for attribute, statistics in (
        attribute_statistics.items()
    ):

        present = statistics["present"]
        absent = statistics["absent"]
        unknown = statistics["unknown"]

        print(f"\n{attribute}")

        print(f"  Present : {present}")

        print(f"  Absent  : {absent}")

        print(f"  Unknown : {unknown}")

        total = present + absent + unknown

        if total != total_images:
            raise ValueError(
                f"Statistics mismatch for "
                f"'{attribute}'."
            )


# Main 

if __name__ == "__main__":

    print(
        "Attributes in DeepFasion Dataset"
        "\n=========================================="
    )
    #Check attribute file

    print("\nCheck for attribute definitions")

    all_attributes = (
        read_attribute_definitions(
            ATTRIBUTE_DEFINITION_FILE
        )
    )

    print(
        f"  Found attributes    : "
        f"{len(all_attributes)}"
    )

    # get selected attributes
    attribute_positions = (
        find_selected_attributes(
            all_attributes,
            SELECTED_ATTRIBUTES
        )
    )

    print(
        f"  Selected attributes : "
        f"{len(SELECTED_ATTRIBUTES)}"
    )

    display_selected_attributes(
        attribute_positions
    )

    #annotations
    print("\nGetting the annotations")

    (
        image_count,
        attribute_statistics
    ) = validate_image_annotations(
        IMAGE_ATTRIBUTE_FILE,
        all_attributes,
        attribute_positions,
        IMAGE_ROOT
    )

    print(f"\n  Expected images : {image_count}")

    print(f"  Processed images: {image_count}")

    print("  Label count per image: 1000")

    print("  Valid label values: -1, 0, 1")


    #Display each attributes statistics

    display_attribute_statistics(attribute_statistics, image_count)

    # result

    print(
        "\n=========================================="
    )

    print(
        "Summary"
    )


    print(
        "=========================================="
    )

    print(
        f"\nImages validated   : {image_count}"
    )

    print(
        f"Total attributes   : {len(all_attributes)}"
    )

    print(
        f"Selected attributes: "
        f"{len(SELECTED_ATTRIBUTES)}"
    )