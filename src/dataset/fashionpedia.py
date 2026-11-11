"""
fashionpedia.py   ->  save as  src/dataset/fashionpedia.py

Reads the Fashionpedia annotation file and turns it into attribute records
that look like the DeepFashion ones, plus a MASK.

Why a mask: Fashionpedia only tells us about three of our 26 attributes
(collar, pocket, zipper). It says nothing about "red" or "floral". A mask
marks which of the 26 label slots an image really has an answer for, so the
training loss can ignore the unknown ones.

Each record is:  (image_path, labels, mask)
    labels : 26 numbers, 1.0 = part present, 0.0 = absent
    mask   : 26 numbers, 1.0 = this slot is known, 0.0 = unknown (ignore it)
"""

import json
import os

from src.dataset.attributemanage import SELECTED_ATTRIBUTES


# ============================================================
# SETTINGS - change these paths to where you saved the files
# ============================================================

FASHIONPEDIA_ROOT = r"C:\Users\User\crisp\dataset\fashionpedia"

TRAIN_ANNOTATIONS = os.path.join(
    FASHIONPEDIA_ROOT, "instances_attributes_train2020.json"
)
TRAIN_IMAGES = os.path.join(FASHIONPEDIA_ROOT, "train")

VAL_ANNOTATIONS = os.path.join(
    FASHIONPEDIA_ROOT, "instances_attributes_val2020.json"
)
VAL_IMAGES = os.path.join(FASHIONPEDIA_ROOT, "val_test")

# Fashionpedia category id -> the name used in SELECTED_ATTRIBUTES.
# The ids come from the categories list printed from the annotation file:
#   28 = collar, 32 = pocket, 35 = zipper
PART_TO_ATTRIBUTE = {
    28: "collar",
    32: "pocket",
    35: "zipper",
}


# ============================================================
# HELPERS
# ============================================================

def find_image_files(image_dir):
    """
    Walk through the image folder once and remember where every file is.

    Why: after unzipping, the images can sit directly in the folder or one
    level deeper (train/train/...). Looking files up by name works for both.
    """
    lookup = {}

    for folder, _, file_names in os.walk(image_dir):
        for file_name in file_names:
            lookup[file_name] = os.path.join(folder, file_name)

    return lookup


# ============================================================
# BUILD RECORDS
# ============================================================

def build_fashionpedia_records(annotation_file, image_dir):
    """
    For every image, decide which of our three part attributes are present.

    Steps:
      1. Read the annotation file (it is large, so this takes a minute).
      2. For each annotated object, if its category is collar, pocket or
         zipper, remember "this image has that part".
      3. Build one record per image: labels for the 26 slots, plus a mask
         that switches on only the three slots Fashionpedia knows about.
    """

    number_of_attributes = len(SELECTED_ATTRIBUTES)

    # Position of each part inside the 26 label slots
    slot_of = {
        name: SELECTED_ATTRIBUTES.index(name)
        for name in PART_TO_ATTRIBUTE.values()
    }

    # The mask is the same for every Fashionpedia image
    known_mask = [0.0] * number_of_attributes
    for slot in slot_of.values():
        known_mask[slot] = 1.0

    print(f"Reading Fashionpedia annotations: {annotation_file}")

    with open(annotation_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # image id -> set of part names found in that image
    parts_in_image = {}

    for annotation in data["annotations"]:

        part_name = PART_TO_ATTRIBUTE.get(annotation["category_id"])

        if part_name is None:
            continue

        parts_in_image.setdefault(
            annotation["image_id"], set()
        ).add(part_name)

    file_lookup = find_image_files(image_dir)

    records = []
    missing_files = 0

    for image in data["images"]:

        image_path = file_lookup.get(
            os.path.basename(image["file_name"])
        )

        if image_path is None:
            missing_files += 1
            continue

        labels = [0.0] * number_of_attributes

        for part_name in parts_in_image.get(image["id"], ()):
            labels[slot_of[part_name]] = 1.0

        records.append((image_path, labels, list(known_mask)))

    print(f"Fashionpedia images used: {len(records)}")

    if missing_files > 0:
        print(
            f"Warning: {missing_files} images listed in the annotation file "
            f"were not found in {image_dir}"
        )

    # Show how many images contain each part
    for name, slot in slot_of.items():
        count = sum(1 for record in records if record[1][slot] == 1.0)
        print(f"  {name:<8}: {count} images")

    return records


# ============================================================
# QUICK CHECK:  python -m src.dataset.fashionpedia
# ============================================================

if __name__ == "__main__":

    records = build_fashionpedia_records(VAL_ANNOTATIONS, VAL_IMAGES)

    path, labels, mask = records[0]
    print("\nFirst record:")
    print("  image :", path)
    print("  labels:", [
        SELECTED_ATTRIBUTES[i] for i, v in enumerate(labels) if v == 1.0
    ])
    print("  known :", [
        SELECTED_ATTRIBUTES[i] for i, v in enumerate(mask) if v == 1.0
    ])