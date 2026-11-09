
from collections import Counter
from pathlib import Path

file_path = input("Enter the full path to list_attr_img.txt: ").strip().strip('"')

label_counts = Counter()
row_lengths = Counter()
data_rows = 0
examples = []

with open(file_path, "r", encoding="utf-8") as file:
    declared_count = file.readline().strip()
    header = file.readline().strip()

    for line in file:
        parts = line.split()

        if len(parts) < 2:
            continue

        image_name = parts[0]
        labels = parts[1:]

        data_rows += 1
        row_lengths[len(labels)] += 1
        label_counts.update(labels)

        if len(examples) < 5:
            positive_positions = [
                i for i, value in enumerate(labels)
                if value == "1"
            ]
            examples.append(
                (image_name, len(labels), positive_positions)
            )

print("\n=== LABEL VALUE COUNTS ===")
print("Declared image count:", declared_count)
print("Data rows read:", data_rows)
print("Label values found:", dict(label_counts))
print("Number of labels per row:", dict(row_lengths))

print("\n=== FIRST 5 IMAGE ROWS ===")
for image, count, positives in examples:
    print("\nImage:", image)
    print("Number of labels:", count)
    print("Positive label positions (zero-based):", positives[:30])
