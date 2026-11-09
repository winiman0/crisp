"""
Label meaning check (read-only: no training, no model changes).

Save as src/evaluation/label_name_check.py and run:

    python -m src.evaluation.label_name_check

Idea: DeepFashion product folders are named after the garment, for example
"Sheer_Pleated-Front_Blouse" or "Floral_Print_Chiffon_Dress". If the label
column for "floral" really means floral, then images whose folder name
contains "floral" should be labelled floral far more often than images in
general. If the column were pointing at the wrong attribute, the two rates
would be about the same.

Reading the table:
    In-title rate   share of images with the keyword in the folder name
                    that carry the label
    Overall rate    share of ALL images that carry the label
    Lift            in-title rate divided by overall rate
A lift well above 1 (say 3 or more) means the label column matches the name.
A lift near 1 means the column is probably not what its name says.
"""

import os

from src.preprocessing.trainpre2 import build_attribute_records
from src.dataset.attributemanage import SELECTED_ATTRIBUTES


# Word to look for in the product folder name, per attribute.
# Attributes that are rarely written in product titles are left out.
TITLE_KEYWORDS = {
    "button": "button",
    "zipper": "zip",
    "collar": "collar",
    "pocket": "pocket",
    "red": "red",
    "pink": "pink",
    "floral": "floral",
    "striped": "stripe",
    "plaid": "plaid",
    "polka dot": "polka",
    "sleeveless": "sleeveless",
    "crew neck": "crew",
    "v-neck": "v-neck",
    "boat neck": "boat",
    "scoop-neck": "scoop",
    "turtle-neck": "turtle",
    "cotton": "cotton",
    "denim": "denim",
    "chiffon": "chiffon",
    "leather": "leather",
    "knit": "knit",
    "lace": "lace",
}


def run_check():
    print("Reading annotations...")
    records = build_attribute_records()

    # Folder name of each image, lower case
    folders = [
        os.path.basename(os.path.dirname(path)).lower()
        for path, _ in records
    ]

    total = len(records)

    print(f"\nImages: {total}")
    print("\nLabel meaning check")
    print("=" * 78)
    print(
        f"{'Attribute':<14}{'Keyword':<12}{'Images with':>12}"
        f"{'In-title':>11}{'Overall':>10}{'Lift':>8}   Verdict"
    )
    print(f"{'':<26}{'keyword':>12}{'rate':>11}{'rate':>10}")
    print("-" * 78)

    for attribute, keyword in TITLE_KEYWORDS.items():
        if attribute not in SELECTED_ATTRIBUTES:
            continue

        column = SELECTED_ATTRIBUTES.index(attribute)

        overall_positive = 0
        title_count = 0
        title_positive = 0

        for folder, (_, labels) in zip(folders, records):
            has_label = labels[column] == 1.0

            if has_label:
                overall_positive += 1

            if keyword in folder:
                title_count += 1
                if has_label:
                    title_positive += 1

        overall_rate = overall_positive / total

        if title_count < 30:
            print(
                f"{attribute:<14}{keyword:<12}{title_count:>12}"
                f"{'-':>11}{overall_rate:>10.3f}{'-':>8}   too few titles to judge"
            )
            continue

        title_rate = title_positive / title_count
        lift = title_rate / overall_rate if overall_rate > 0 else 0.0

        if lift >= 3.0:
            verdict = "matches"
        elif lift >= 1.5:
            verdict = "weak match"
        else:
            verdict = "DOES NOT MATCH"

        print(
            f"{attribute:<14}{keyword:<12}{title_count:>12}"
            f"{title_rate:>11.3f}{overall_rate:>10.3f}{lift:>8.1f}   {verdict}"
        )


if __name__ == "__main__":
    run_check()