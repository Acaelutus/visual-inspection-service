import cv2
import shutil
import numpy as np
from pathlib import Path

CLASS_ID = 0
CLASS_NAME = "defect"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def mask_to_bboxes(mask_path: Path, image_size: tuple) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    img_w, img_h = image_size
    yolo_lines = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        yolo_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w/img_w:.6f} {h/img_h:.6f}")

    return yolo_lines


def process_category(category_path: Path, output_path: Path, val_ratio: float = 0.2) -> dict:
    stats = {"processed": 0, "skipped": 0, "category": category_path.name}

    test_dir = category_path / "test"
    gt_dir = category_path / "ground_truth"

    if not test_dir.exists():
        return stats

    defect_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]

    all_images = []
    for defect_dir in defect_dirs:
        for img_path in sorted(defect_dir.glob("*.png")):
            mask_path = gt_dir / defect_dir.name / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                all_images.append((img_path, mask_path))

    for idx, (img_path, mask_path) in enumerate(all_images):
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            continue

        h, w = img.shape[:2]
        yolo_lines = mask_to_bboxes(mask_path, (w, h))

        if not yolo_lines:
            stats["skipped"] += 1
            continue

        split = "val" if idx % int(1 / val_ratio) == 0 else "train"
        unique_name = f"{category_path.name}_{img_path.parent.name}_{img_path.stem}"

        shutil.copy2(img_path, output_path / "images" / split / f"{unique_name}.png")
        (output_path / "labels" / split / f"{unique_name}.txt").write_text("\n".join(yolo_lines))

        stats["processed"] += 1

    return stats


def prepare_full_dataset(mvtec_path: str, output_path: str):
    mvtec = Path(mvtec_path)
    output = Path(output_path)

    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_skipped = 0

    for category_name in CATEGORIES:
        category_path = mvtec / category_name
        if not category_path.exists():
            print(f"✗ Not found: {category_name}")
            continue

        print(f"Processing: {category_name}...")
        stats = process_category(category_path, output)
        print(f"  ✓ {stats['processed']} processed, {stats['skipped']} skipped")
        total_processed += stats["processed"]
        total_skipped += stats["skipped"]

    yaml_content = f"path: {output}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: {CLASS_NAME}\n"
    (output / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    train_count = len(list((output / "images" / "train").glob("*.png")))
    val_count = len(list((output / "images" / "val").glob("*.png")))

    print(f"\n{'='*50}")
    print(f"✓ Done! {total_processed} processed, {total_skipped} skipped")
    print(f"  Train: {train_count} | Val: {val_count}")
    print(f"  Output: {output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    prepare_full_dataset(
        mvtec_path=r"D:\mvtec_anomaly_detection",
        output_path=r"D:\visual-inspection-service\data\mvtec_yolo"
    )
