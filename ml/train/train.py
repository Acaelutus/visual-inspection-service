"""
Локальное обучение YOLOv8 на датасете MVTec.

Для полноценного обучения используй kaggle_train.py (GPU T4).
Этот скрипт — для быстрой проверки что всё работает локально (CPU).
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Пути по умолчанию — относительно корня репозитория
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "mvtec_yolo" / "dataset.yaml"
DEFAULT_RUNS = ROOT / "ml" / "runs"


def train(data: Path, runs: Path, epochs: int, batch: int):
    """Запускает обучение YOLOv8."""
    # yolov8n = nano — самая маленькая, быстро проверить pipeline
    # yolov8s = small — для реального обучения
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="mvtec_local",
        project=str(runs),
        patience=10,
        save=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✓ Готово! Лучшая модель: {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 локально")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Путь до dataset.yaml"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUNS,
        help="Папка для сохранения результатов"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,         # локально делаем 5 эпох — просто smoke test
        help="Количество эпох"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,         # CPU: маленький batch чтобы не упасть по памяти
        help="Размер батча"
    )
    args = parser.parse_args()

    train(args.data, args.runs, args.epochs, args.batch)
