"""
Celery задачи — функции, которые выполняются в воркере асинхронно.

Воркер — отдельный процесс (python process), который:
1. Подключается к Redis и ждёт задач в очереди
2. Берёт задачу
3. Выполняет inference
4. Сохраняет результат в Redis (backend)
"""

import base64

import cv2
import numpy as np
from celery.signals import worker_ready

from app.core.celery_app import celery_app
from app.core.model import get_detector, load_model


@worker_ready.connect
def on_worker_ready(**kwargs):
    """
    Celery сигнал — вызывается один раз когда воркер поднялся.
    Загружаем модель здесь, чтобы не грузить её при каждой задаче.
    """
    print("Воркер запущен, загружаем модель...")
    load_model()
    print("Модель загружена, воркер готов к работе")


@celery_app.task(name="tasks.detect_defects", bind=True)
def detect_defects(self, image_b64: str, confidence: float = 0.25) -> dict:
    """
    Задача детекции дефектов.

    bind=True — self ссылается на саму задачу, нужен для self.retry().

    Почему image_b64 строка, а не bytes?
    Celery по умолчанию сериализует задачи в JSON.
    JSON не умеет bytes, поэтому кодируем изображение в base64 строку.

    Args:
        image_b64: изображение в base64 (строка, JSON-сериализуемо)
        confidence: порог уверенности детектора

    Returns:
        dict с результатами (JSON-сериализуемый, сохраняется в Redis)
    """
    try:
        # base64 строка → bytes → numpy array → OpenCV изображение BGR
        raw = base64.b64decode(image_b64)
        buf = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Не удалось декодировать изображение в воркере")

        detector = get_detector()
        defects, inference_ms = detector.predict(image, confidence=confidence)

        # Pydantic объекты → dict, чтобы можно было сохранить в JSON/Redis
        return {
            "defects": [d.model_dump() for d in defects],
            "count": len(defects),
            "inference_ms": inference_ms,
        }

    except Exception as exc:
        # При ошибке — повторная попытка через 5 секунд, максимум 3 раза
        raise self.retry(exc=exc, countdown=5, max_retries=3)
