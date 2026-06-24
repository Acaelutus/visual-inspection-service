import base64

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.celery_app import celery_app
from app.core.metrics import DEFECTS_DETECTED_TOTAL, INFERENCE_DURATION_MS, PREDICTIONS_TOTAL
from app.models.schemas import TaskResponse, TaskStatus
from app.workers.tasks import detect_defects

router = APIRouter(prefix="/tasks", tags=["tasks"])

ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")


@router.post("/predict", response_model=TaskResponse)
async def create_predict_task(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    raw = await file.read()
    task = detect_defects.delay(base64.b64encode(raw).decode("utf-8"), confidence)

    return TaskResponse(task_id=task.id, status=TaskStatus.pending)


@router.get("/{task_id}", response_model=TaskResponse)
def get_task_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "SUCCESS":
        data = result.result
        # forget() prevents double-counting metrics on repeated polls
        result.forget()
        PREDICTIONS_TOTAL.labels(endpoint="async").inc()
        INFERENCE_DURATION_MS.observe(data["inference_ms"])
        DEFECTS_DETECTED_TOTAL.inc(data["count"])
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.success,
            defects=data["defects"],
            count=data["count"],
            inference_ms=data["inference_ms"],
        )

    if result.state == "FAILURE":
        return TaskResponse(task_id=task_id, status=TaskStatus.failure, error=str(result.result))

    return TaskResponse(task_id=task_id, status=TaskStatus.pending)
