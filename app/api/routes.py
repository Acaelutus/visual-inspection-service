import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.core.metrics import DEFECTS_DETECTED_TOTAL, INFERENCE_DURATION_MS, PREDICTIONS_TOTAL
from app.core.model import DefectDetector, get_detector, is_model_loaded
from app.models.schemas import HealthResponse, PredictResponse

router = APIRouter()

ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=is_model_loaded())


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0),
    detector: DefectDetector = Depends(get_detector),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    defects, inference_ms = detector.predict(image, confidence=confidence)

    PREDICTIONS_TOTAL.labels(endpoint="sync").inc()
    INFERENCE_DURATION_MS.observe(inference_ms)
    DEFECTS_DETECTED_TOTAL.inc(len(defects))

    return PredictResponse(defects=defects, count=len(defects), inference_ms=inference_ms)
