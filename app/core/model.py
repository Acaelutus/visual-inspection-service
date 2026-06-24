"""
Загрузка и запуск YOLO модели.

Паттерн Singleton: модель загружается один раз при старте сервиса
и переиспользуется для всех запросов. Загрузка занимает ~1-2 секунды,
поэтому делать это при каждом запросе — недопустимо для production.
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

from app.models.schemas import BBox, Defect

# Путь до модели — два уровня вверх от app/core/
MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "models" / "best.pt"


class DefectDetector:
    """
    Обёртка над YOLO моделью.
    Один экземпляр на весь сервис — создаётся в lifespan FastAPI.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {MODEL_PATH}\n"
                f"Положи best.pt в ml/models/"
            )
        # task='detect' — явно указываем тип задачи
        self.model = YOLO(str(MODEL_PATH), task="detect")
        self.model_path = MODEL_PATH
        print(f"✓ Модель загружена: {MODEL_PATH.name}")

    def predict(self, image: np.ndarray, confidence: float = 0.25) -> tuple[list[Defect], float]:
        """
        Запускает детекцию на изображении.

        Args:
            image: изображение в формате BGR numpy array (как из cv2)
            confidence: порог уверенности — детекции ниже этого порога игнорируются

        Returns:
            (список дефектов, время inference в мс)
        """
        start = time.perf_counter()

        # verbose=False — не засорять лог сервера выводом YOLO
        results = self.model.predict(source=image, conf=confidence, verbose=False)

        elapsed_ms = (time.perf_counter() - start) * 1000

        defects = []
        result = results[0]  # один результат — одно изображение

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            defects.append(Defect(
                class_name=self.model.names[cls_id],
                confidence=round(conf, 4),
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            ))

        return defects, round(elapsed_ms, 2)


# Глобальный экземпляр — None до момента загрузки при старте сервиса
_detector: Optional[DefectDetector] = None


def get_detector() -> DefectDetector:
    """
    FastAPI Dependency — возвращает готовый детектор.
    Вызывается автоматически при каждом запросе к /predict.
    """
    if _detector is None:
        raise RuntimeError("Детектор не инициализирован — сервис ещё не запущен")
    return _detector


def load_model() -> None:
    """Вызывается один раз при старте сервиса (в lifespan)."""
    global _detector
    _detector = DefectDetector()


def is_model_loaded() -> bool:
    return _detector is not None
