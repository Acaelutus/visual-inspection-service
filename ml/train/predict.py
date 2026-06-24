"""
Запуск inference на одном изображении.
Проверяем что модель реально находит дефекты.
"""

import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path

# Корень репозитория — два уровня вверх от этого файла
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "ml" / "models" / "best.pt"


def predict_single(image_path: str, model_path: str, confidence: float = 0.25):
    """
    Запускает модель на одном изображении и показывает результат.

    Args:
        image_path: путь до изображения
        model_path: путь до обученной модели best.pt
        confidence: минимальная уверенность для детекции (0-1)
    """
    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        conf=confidence,
        save=False,
        verbose=False
    )

    result = results[0]

    # Отладка — сколько объектов нашла модель до фильтрации по conf
    print(f"Найдено boxes: {len(result.boxes)}")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        print(f"  → {cls_name}: уверенность {conf:.3f}")

    img = result.orig_img.copy()

    if len(result.boxes) == 0:
        print("✓ Дефектов не обнаружено")
        cv2.imshow("Prediction", cv2.resize(img, (600, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Рисуем каждый найденный дефект
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]

        # Координаты bbox в пикселях (xyxy = x1,y1,x2,y2 — два угла прямоугольника)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print(f"Дефект: {cls_name} (conf={conf:.2f}) @ ({x1},{y1})-({x2},{y2})")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(
            img, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

    cv2.imshow("Prediction — Visual Inspection", cv2.resize(img, (600, 600)))
    print("\nНажми любую клавишу чтобы закрыть...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference YOLOv8 на изображении")
    parser.add_argument("image", type=str, help="Путь до изображения")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Путь до best.pt (по умолчанию: ml/models/best.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Минимальная уверенность детекции (0-1)"
    )
    args = parser.parse_args()

    predict_single(args.image, args.model, args.conf)
