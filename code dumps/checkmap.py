
from pathlib import Path

base = Path(
    r"C:\Users\User\crisp\dataset\Category and Attribute Prediction Benchmark\Anno_coarse"
)

attributes_file = base / "list_attr_cloth.txt"

selected_attributes = [
    "button", "zipper", "collar", "pocket",
    "red", "pink", "dark", "light",
    "floral", "striped", "plaid", "polka dot",
    "long sleeve", "long-sleeved", "sleeveless",
    "crew neck", "v-neck", "boat neck", "scoop-neck",
    "turtle-neck", "cotton", "denim", "chiffon",
    "leather", "knit", "lace",
]

with open(attributes_file, "r", encoding="utf-8") as file:
    file.readline()  # Number of attributes
    file.readline()  # Header
    attribute_lines = file.readlines()

attribute_names = [
    line.split(maxsplit=1)[0].strip().lower()
    for line in attribute_lines
]

print("Selected attribute mapping")
print("=" * 55)

for name in selected_attributes:
    matches = [
        i for i, attribute in enumerate(attribute_names)
        if attribute == name.lower()
    ]

    if matches:
        print(f"{name:18} -> index {matches[0]}")
    else:
        print(f"{name:18} -> NOT FOUND")
