"""
Скрипт обучения YOLOv8 на Kaggle GPU (Tesla T4).

Как использовать:
  1. Загрузить data/mvtec_yolo.zip на Kaggle как новый Dataset
  2. Создать новый Kaggle Notebook (+ New Notebook)
  3. В Settings → Accelerator выбрать GPU T4 x2
  4. Добавить датасет через Add Data → Your Datasets
  5. Вставить содержимое этого файла в notebook
  6. Run All

Датасет будет доступен по пути /kaggle/input/<slug>/mvtec_yolo/
где <slug> — это имя датасета которое ты задал при загрузке.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
# Измени этот slug на имя своего Kaggle датасета
DATASET_SLUG = "mvtec-yolo-defects"

# Kaggle всегда монтирует датасеты сюда
KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")

# ─── НАХОДИМ ДАТАСЕТ ──────────────────────────────────────────────────────────
# Ищем папку mvtec_yolo внутри примонтированного датасета
dataset_root = KAGGLE_INPUT / DATASET_SLUG / "mvtec_yolo"

# Если путь неверный — показываем что доступно, чтобы было легче найти
if not dataset_root.exists():
    print("Доступные датасеты:")
    for p in KAGGLE_INPUT.iterdir():
        print(f"  {p}")
        for sub in p.iterdir():
            print(f"    {sub}")
    raise FileNotFoundError(
        f"Датасет не найден: {dataset_root}\n"
        f"Измени DATASET_SLUG на правильное имя выше."
    )

print(f"✓ Датасет найден: {dataset_root}")

# ─── СТАТИСТИКА ДАТАСЕТА ──────────────────────────────────────────────────────
train_images = list((dataset_root / "images" / "train").glob("*.png"))
val_images = list((dataset_root / "images" / "val").glob("*.png"))
print(f"  Train: {len(train_images)} изображений")
print(f"  Val:   {len(val_images)} изображений")

# ─── ПАТЧИМ dataset.yaml ──────────────────────────────────────────────────────
# Оригинальный yaml содержит Windows путь D:\...
# Заменяем на актуальный Kaggle путь
yaml_src = dataset_root / "dataset.yaml"
yaml_dst = WORK_DIR / "dataset.yaml"

with open(yaml_src) as f:
    config = yaml.safe_load(f)

# path: корневая папка датасета — от неё строятся train: и val:
config["path"] = str(dataset_root)

with open(yaml_dst, "w") as f:
    yaml.dump(config, f, allow_unicode=True)

print(f"\n✓ dataset.yaml создан: {yaml_dst}")
print(f"  path: {config['path']}")
print(f"  nc:   {config['nc']}")
print(f"  names: {config['names']}")

# ─── ПРОВЕРКА GPU ─────────────────────────────────────────────────────────────
import torch
print(f"\n GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── ОБУЧЕНИЕ ─────────────────────────────────────────────────────────────────
# YOLOv8s (small) — хороший баланс точности и скорости для T4
# Размеры: n=nano, s=small, m=medium, l=large, x=extra-large
# На T4 (~16GB VRAM) комфортно: yolov8s при batch=16
model = YOLO("yolov8s.pt")

results = model.train(
    data=str(yaml_dst),
    epochs=100,          # 100 эпох — достаточно для fine-tuning
    imgsz=640,           # стандартный размер для YOLOv8
    batch=16,            # T4 тянет batch=16 при 640px
    name="mvtec_v1",
    project=str(WORK_DIR / "runs"),
    patience=20,         # остановка если 20 эпох без улучшения
    save=True,
    save_period=10,      # сохраняем checkpoint каждые 10 эпох
    verbose=True,
    # ─── Аугментации ───────────────────────────────────────────────
    # Для маленького датасета (1258 изображений) агрессивная аугментация
    # помогает модели не переобучиться
    hsv_h=0.015,         # случайный сдвиг оттенка
    hsv_s=0.7,           # случайное насыщение
    hsv_v=0.4,           # случайная яркость
    degrees=10.0,        # случайный поворот ±10°
    translate=0.1,       # случайный сдвиг ±10% изображения
    scale=0.5,           # случайный масштаб ±50%
    flipud=0.3,          # вертикальное отражение с P=0.3
    fliplr=0.5,          # горизонтальное отражение с P=0.5
    mosaic=1.0,          # mosaic аугментация (склеивает 4 изображения)
)

# ─── РЕЗУЛЬТАТЫ ───────────────────────────────────────────────────────────────
best_model = Path(results.save_dir) / "weights" / "best.pt"
metrics = results.results_dict

print(f"\n{'='*50}")
print(f"✓ Обучение завершено!")
print(f"  Лучшая модель: {best_model}")
print(f"  mAP50:    {metrics.get('metrics/mAP50(B)', 0):.4f}")
print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")
print(f"{'='*50}")
print(f"\nМодель сохранена в Output — скачай через Kaggle UI")
