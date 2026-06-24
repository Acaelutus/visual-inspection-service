import yaml
from pathlib import Path
from ultralytics import YOLO

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")


def find_dataset_root(base: Path) -> Path:
    for yaml_file in base.rglob("dataset.yaml"):
        return yaml_file.parent
    return None


dataset_root = find_dataset_root(KAGGLE_INPUT)

if dataset_root is None:
    print("dataset.yaml not found. /kaggle/input structure:")
    for p in KAGGLE_INPUT.rglob("*"):
        depth = len(p.relative_to(KAGGLE_INPUT).parts)
        if depth <= 3:
            print("  " * (depth - 1) + p.name + ("/" if p.is_dir() else ""))
    raise FileNotFoundError("dataset.yaml not found in /kaggle/input")

print(f"✓ Dataset: {dataset_root}")

train_images = list((dataset_root / "images" / "train").glob("*.png"))
val_images = list((dataset_root / "images" / "val").glob("*.png"))
print(f"  Train: {len(train_images)} images")
print(f"  Val:   {len(val_images)} images")

yaml_dst = WORK_DIR / "dataset.yaml"
config = {
    "path": str(dataset_root),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": {0: "defect"},
}

with open(yaml_dst, "w", encoding="utf-8") as f:
    yaml.dump(config, f, allow_unicode=True)

import torch
gpu_ok = torch.cuda.is_available()
print(f"\nGPU: {gpu_ok}")
if gpu_ok:
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = YOLO("yolov8s.pt")

results = model.train(
    data=str(yaml_dst),
    epochs=100,
    imgsz=640,
    batch=16,
    name="mvtec_v1",
    project=str(WORK_DIR / "runs"),
    patience=20,
    save=True,
    save_period=10,
    verbose=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.3,
    fliplr=0.5,
    mosaic=1.0,
)

best_model = Path(results.save_dir) / "weights" / "best.pt"
metrics = results.results_dict

print(f"\n{'='*50}")
print(f"✓ Training complete!")
print(f"  Best model: {best_model}")
print(f"  mAP50:      {metrics.get('metrics/mAP50(B)', 0):.4f}")
print(f"  mAP50-95:   {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
print(f"  Precision:  {metrics.get('metrics/precision(B)', 0):.4f}")
print(f"  Recall:     {metrics.get('metrics/recall(B)', 0):.4f}")
print(f"{'='*50}")
