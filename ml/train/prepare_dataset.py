"""
Конвертация MVTec AD масок в YOLO формат.

MVTec даёт нам маски (чёрно-белые PNG где белое = дефект).
YOLO ожидает txt файлы с координатами: класс cx cy width height
Этот скрипт конвертирует одно в другое.
"""

import cv2
import numpy as np
from pathlib import Path


# Классы дефектов — каждому присваиваем номер
# YOLO не понимает названия, только цифры
DEFECT_CLASSES = {
    "broken_large": 0,
    "broken_small": 1,
    "contamination": 2,
}


def mask_to_bbox_yolo(mask_path: Path, image_size: tuple) -> list[str]:
    """
    Читает маску дефекта и возвращает bbox в YOLO формате.

    YOLO формат: класс cx cy width height
    Все значения от 0 до 1 (доли от размера картинки).

    Args:
        mask_path: путь до маски PNG
        image_size: (width, height) оригинального изображения

    Returns:
        список строк в YOLO формате
    """
    # Читаем маску как чёрно-белое изображение
    # cv2.IMREAD_GRAYSCALE — загружает как серый, не цветной
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Не могу прочитать маску: {mask_path}")
        return []

    # Бинаризация — делаем пиксели либо 0 либо 255
    # Всё что > 128 становится 255 (белое = дефект)
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Находим контуры белых областей
    # contours — это список координат границ каждого белого пятна
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,      # только внешние контуры
        cv2.CHAIN_APPROX_SIMPLE # сжатый формат
    )

    if not contours:
        return []

    img_w, img_h = image_size
    yolo_lines = []

    # Определяем класс из имени папки маски
    # mask_path.parent.name = "broken_large" например
    defect_type = mask_path.parent.name
    class_id = DEFECT_CLASSES.get(defect_type, 0)

    for contour in contours:
        # boundingRect даёт нам прямоугольник вокруг контура
        # x, y — левый верхний угол, w, h — ширина и высота
        x, y, w, h = cv2.boundingRect(contour)

        # Конвертируем в YOLO формат (доли от размера картинки)
        # YOLO хочет центр прямоугольника, не угол
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def prepare_dataset(mvtec_path: str, output_path: str):
    """
    Основная функция — конвертирует весь датасет.

    Args:
        mvtec_path: путь до папки bottle из MVTec
        output_path: куда сохранить готовый датасет для YOLO
    """
    mvtec = Path(mvtec_path)
    output = Path(output_path)

    # Создаём структуру папок для YOLO
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    # Проходим по каждому типу дефекта
    for defect_type in DEFECT_CLASSES:
        test_dir = mvtec / "test" / defect_type
        mask_dir = mvtec / "ground_truth" / defect_type

        if not test_dir.exists():
            print(f"Папка не найдена: {test_dir}")
            continue

        images = sorted(test_dir.glob("*.png"))
        print(f"\n{defect_type}: найдено {len(images)} изображений")

        for img_path in images:
            # Находим соответствующую маску
            # 000.png → 000_mask.png
            mask_path = mask_dir / f"{img_path.stem}_mask.png"

            if not mask_path.exists():
                print(f"  Маска не найдена: {mask_path}")
                skipped += 1
                continue

            # Читаем размер изображения
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]  # shape возвращает (height, width, channels)

            # Конвертируем маску в YOLO bbox
            yolo_lines = mask_to_bbox_yolo(mask_path, (w, h))

            if not yolo_lines:
                skipped += 1
                continue

            # 80% train, 20% val — стандартное разделение
            # используем индекс файла для разделения
            idx = int(img_path.stem)
            split = "train" if idx % 5 != 0 else "val"

            # Копируем изображение
            import shutil
            dest_img = output / "images" / split / img_path.name
            shutil.copy2(img_path, dest_img)

            # Сохраняем лейбл
            dest_label = output / "labels" / split / f"{img_path.stem}.txt"
            dest_label.write_text("\n".join(yolo_lines))

            processed += 1

    print(f"\n✓ Готово: {processed} изображений обработано, {skipped} пропущено")
    print(f"✓ Датасет сохранён в: {output}")


if __name__ == "__main__":
    prepare_dataset(
        mvtec_path=r"C:\Users\abduv\Downloads\bottle",
        output_path=r"D:\visual-inspection-service\data\bottle_yolo"
    )