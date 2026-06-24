import base64

import cv2
import numpy as np
from celery.signals import worker_ready

from app.core.celery_app import celery_app
from app.core.model import get_detector, load_model


@worker_ready.connect
def on_worker_ready(**kwargs):
    load_model()


@celery_app.task(name="tasks.detect_defects", bind=True)
def detect_defects(self, image_b64: str, confidence: float = 0.25) -> dict:
    try:
        raw = base64.b64decode(image_b64)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image in worker")

        defects, inference_ms = get_detector().predict(image, confidence=confidence)

        return {
            "defects": [d.model_dump() for d in defects],
            "count": len(defects),
            "inference_ms": inference_ms,
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=5, max_retries=3)
