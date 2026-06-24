"""
Скрипт обучения YOLOv8 на Kaggle GPU (Tesla T4).

Как использовать:
  1. Загрузить data/mvtec_yolo.zip на Kaggle как новый Dataset
  2. Создать новый Kaggle Notebook (+ New Notebook)
  3. В Settings → Accelerator выбрать GPU T4 x2
  4. Добавить датасет через Add Data → Your Datasets
  5. Вставить содержимое этого файла в notebook
  6. Run All
"""

import yaml
from pathlib import Path
from ultralytics import YOLO

# Kaggle всегда монтирует датасеты сюда
KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")

# ─── АВТО-ПОИСК ДАТАСЕТА ──────────────────────────────────────────────────────
# Kaggle распаковывает zip по-разному в зависимости от структуры архива.
# Ищем dataset.yaml — его наличие означает что мы нашли корень датасета.
def find_dataset_root(base: Path) -> Path:
    """Рекурсивно ищет папку с dataset.yaml внутри /kaggle/input."""
    # Проверяем до 3 уровней вглубь — глубже Kaggle обычно не кладёт
    for yaml_file in base.rglob("dataset.yaml"):
        return yaml_file.parent
    return None

dataset_root = find_dataset_root(KAGGLE_INPUT)

# Если не нашли — печатаем дерево для диагностики
if dataset_root is None:
    print("dataset.yaml не найден. Структура /kaggle/input:")
    for p in KAGGLE_INPUT.rglob("*"):
        # Печатаем только первые 3 уровня чтобы не утонуть в списке
        depth = len(p.relative_to(KAGGLE_INPUT).parts)
        if depth <= 3:
            indent = "  " * (depth - 1)
            print(f"{indent}{p.name}{'/' if p.is_dir() else ''}")
    raise FileNotFoundError("dataset.yaml не найден в /kaggle/input")

print(f"✓ Датасет найден: {dataset_root}")

# ─── СТАТИСТИКА ДАТАСЕТА ──────────────────────────────────────────────────────
train_images = list((dataset_root / "images" / "train").glob("*.png"))
val_images = list((dataset_root / "images" / "val").glob("*.png"))
print(f"  Train: {len(train_images)} изображений")
print(f"  Val:   {len(val_images)} изображений")

# ─── СОЗДАЁМ ЧИСТЫЙ dataset.yaml ─────────────────────────────────────────────
# Оригинальный yaml записан с Windows-кодировкой (cp1251) — на Linux он нечитаем.
# Вместо парсинга битого файла создаём новый yaml напрямую — мы знаем его структуру.
yaml_dst = WORK_DIR / "dataset.yaml"

config = {
    "path": str(dataset_root),   # абсолютный путь до корня датасета на Kaggle
    "train": "images/train",     # относительно path
    "val": "images/val",         # относительно path
    "nc": 1,                     # один класс — дефект есть/нет
    "names": {0: "defect"},
}

with open(yaml_dst, "w", encoding="utf-8") as f:
    yaml.dump(config, f, allow_unicode=True)

print(f"\n✓ dataset.yaml создан: {yaml_dst}")
print(f"  path: {config['path']}")
print(f"  nc:   {config['nc']}")
print(f"  names: {config['names']}")

# ─── ПРОВЕРКА GPU ─────────────────────────────────────────────────────────────
import torch
gpu_ok = torch.cuda.is_available()
print(f"\nGPU доступен: {gpu_ok}")
if gpu_ok:
    print(f"  Устройство: {torch.cuda.get_device_name(0)}")
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
