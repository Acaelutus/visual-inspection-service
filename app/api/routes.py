"""
API endpoints сервиса детекции дефектов.
"""

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.core.metrics import DEFECTS_DETECTED_TOTAL, INFERENCE_DURATION_MS, PREDICTIONS_TOTAL
from app.core.model import DefectDetector, get_detector, is_model_loaded
from app.models.schemas import HealthResponse, PredictResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """
    Health check — мониторинг (Prometheus, load balancer) дёргает этот endpoint
    чтобы убедиться что сервис живой и модель загружена.
    """
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Изображение для анализа (PNG, JPG)"),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0, description="Порог уверенности"),
    detector: DefectDetector = Depends(get_detector),
):
    """
    Детекция дефектов на изображении.

    Принимает изображение, возвращает список найденных дефектов с координатами.
    """
    # Валидируем тип файла — принимаем только изображения
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. Нужен image/jpeg или image/png",
        )

    # Читаем байты и декодируем в numpy array (как если бы мы читали файл через cv2.imread)
    raw = await file.read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="Не удалось декодировать изображение")

    defects, inference_ms = detector.predict(image, confidence=confidence)

    # Обновляем Prometheus метрики после каждого успешного предсказания
    PREDICTIONS_TOTAL.labels(endpoint="sync").inc()
    INFERENCE_DURATION_MS.observe(inference_ms)
    DEFECTS_DETECTED_TOTAL.inc(len(defects))

    return PredictResponse(
        defects=defects,
        count=len(defects),
        inference_ms=inference_ms,
    )
