"""
Обучение YOLOv8 на датасете дефектов бутылок.
Используем fine-tuning — берём pretrained модель и дообучаем.
"""

from ultralytics import YOLO
from pathlib import Path


def train():
    # Загружаем pretrained YOLOv8n (nano — самая маленькая и быстрая)
    # При первом запуске автоматически скачает веса (~6MB)
    # n = nano, s = small, m = medium, l = large, x = extra large
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=r"D:\visual-inspection-service\ml\train\dataset.yaml",
        epochs=50,           # сколько раз пройти по всему датасету
        imgsz=640,           # размер входного изображения
        batch=8,             # сколько картинок обрабатывать за раз
        name="bottle_defects",  # имя эксперимента
        project=r"D:\visual-inspection-service\ml\runs",  # куда сохранять
        patience=10,         # остановить если 10 эпох без улучшения
        save=True,           # сохранять лучшую модель
        verbose=True,        # показывать прогресс
    )

    print(f"\n✓ Обучение завершено")
    print(f"✓ Лучшая модель: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()