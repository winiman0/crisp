import os
import shutil

# file paths
SOURCE_DS = r"C:\Users\User\crisp\dataset\multiclass_dataset\Dataset"
RELABELED_DS = r"C:\Users\User\crisp\dataset\relabeled_cond"

CLASS_MAP = {
    "defect free": "clean",
    "defect-free": "clean", 
    "stain": "stain",
    "hole": "defect",
    "horizontal": "defect",
    "vertical": "defect",
    "lines": "defect",
    "broken stitch": "defect",
    "needle mark": "defect",
    "pinched fabric": "defect",
}


def relabel_dataset(src_path, target_path):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)

    for class_name in os.listdir(src_path):
        class_folder = os.path.join(src_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        key = class_name.lower().strip()
        if key not in CLASS_MAP:
            raise ValueError(
                f"Unrecognized class folder '{class_name}' in multiclass_dataset "
                f"at {class_folder}. Add it to CLASS_MAP explicitly before continuing."
            )

        target_class = CLASS_MAP[key]
        target_class_folder = os.path.join(target_path, target_class)
        os.makedirs(target_class_folder, exist_ok=True)

        for img_file in os.listdir(class_folder):
            src_img = os.path.join(class_folder, img_file)
            dst_img = os.path.join(target_class_folder, img_file)
            if not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)


def compute_class_weights_manual(class_counts: dict):
    n_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    return {
        cls: round(n_samples / (n_classes * count), 4)
        for cls, count in class_counts.items()
    }


if __name__ == "__main__":
    relabel_dataset(SOURCE_DS, RELABELED_DS)

    print("Process complete.\n")
    print("Class :")
    class_counts = {}
    for cls in ["clean", "stain", "defect"]:
        cls_folder = os.path.join(RELABELED_DS, cls)
        n = len(os.listdir(cls_folder)) if os.path.exists(cls_folder) else 0
        class_counts[cls] = n
        print(f"  {cls}: {n} images")

    weights = compute_class_weights_manual(class_counts)
    print("\nClass weights:", weights)
    print("\nExpected totals : clean=1666, stain=398, "
          "defect= 793")