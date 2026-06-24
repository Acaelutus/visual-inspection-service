"""
Sanity check — визуальная проверка разметки.
Рисуем bbox поверх изображения и смотрим глазами
что он попадает на дефект.
"""

import cv2
import numpy as np
from pathlib import Path


# Цвета для каждого класса (BGR формат — в OpenCV порядок Blue Green Red)
COLORS = {
    0: (0, 0, 255),    # broken_large  → красный
    1: (0, 255, 0),    # broken_small  → зелёный
    2: (255, 0, 0),    # contamination → синий
}

CLASS_NAMES = {
    0: "broken_large",
    1: "broken_small",
    2: "contamination",
}


def draw_yolo_labels(image_path: Path, label_path: Path) -> np.ndarray:
    """
    Рисует bbox из YOLO лейбла поверх изображения.

    Args:
        image_path: путь до изображения
        label_path: путь до txt файла с YOLO разметкой

    Returns:
        изображение с нарисованными bbox
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    lines = label_path.read_text().strip().split("\n")

    for line in lines:
        if not line:
            continue

        # Парсим строку YOLO формата
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        # Конвертируем обратно из долей в пиксели
        # YOLO хранит центр и размер → нам нужны углы для рисования
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = COLORS.get(class_id, (255, 255, 255))
        name = CLASS_NAMES.get(class_id, "unknown")

        # Рисуем прямоугольник
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Подпись над прямоугольником
        cv2.putText(
            img, name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

    return img


def visualize_samples(dataset_path: str, num_samples: int = 6):
    """
    Показывает несколько случайных примеров из датасета.

    Args:
        dataset_path: путь до папки bottle_yolo
        num_samples: сколько картинок показать
    """
    dataset = Path(dataset_path)
    images_dir = dataset / "images" / "train"
    labels_dir = dataset / "labels" / "train"

    image_files = sorted(images_dir.glob("*.png"))[:num_samples]

    results = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"Лейбл не найден: {label_path}")
            continue

        annotated = draw_yolo_labels(img_path, label_path)

        # Уменьшаем для отображения — оригинал большой
        annotated = cv2.resize(annotated, (400, 400))
        results.append(annotated)
        print(f"✓ {img_path.name} — лейбл: {label_path.read_text().strip()}")

    if not results:
        print("Нет изображений для отображения")
        return

    # Склеиваем все картинки в одну сетку
    # np.hstack — склеивает горизонтально
    grid = np.hstack(results)

    cv2.imshow("Sanity Check — bbox visualization", grid)
    print("\nНажми любую клавишу чтобы закрыть окно...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_samples(
        dataset_path=r"D:\visual-inspection-service\data\bottle_yolo",
        num_samples=6
    )