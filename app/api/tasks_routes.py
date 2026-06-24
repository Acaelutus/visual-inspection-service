"""
Асинхронные эндпоинты для работы с очередью задач.

Флоу:
1. POST /tasks/predict → клиент отправляет изображение, получает task_id сразу
2. GET  /tasks/{task_id} → клиент проверяет статус, получает результат когда готов

Это паттерн "fire and forget" + polling.
"""

import base64

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.celery_app import celery_app
from app.models.schemas import TaskResponse, TaskStatus
from app.workers.tasks import detect_defects

# prefix="/tasks" — все маршруты начинаются с /tasks
router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/predict", response_model=TaskResponse)
async def create_predict_task(
    file: UploadFile = File(..., description="Изображение для анализа (PNG, JPG)"),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0, description="Порог уверенности"),
):
    """
    Отправляет задачу детекции в очередь.
    Возвращает task_id немедленно — клиент НЕ ждёт пока модель обработает изображение.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. Нужен image/jpeg или image/png",
        )

    raw = await file.read()

    # bytes → base64 строка, чтобы передать через JSON (Redis/Celery не умеют bytes напрямую)
    image_b64 = base64.b64encode(raw).decode("utf-8")

    # .delay() — отправляет задачу в Redis-очередь и сразу возвращает AsyncResult
    # Это НЕ блокирующий вызов — он не ждёт выполнения задачи
    task = detect_defects.delay(image_b64, confidence)

    return TaskResponse(task_id=task.id, status=TaskStatus.pending)


@router.get("/{task_id}", response_model=TaskResponse)
def get_task_result(task_id: str):
    """
    Проверяет статус задачи и возвращает результат если готов.

    Возможные состояния Celery:
    - PENDING  : задача в очереди или task_id не существует
    - STARTED  : воркер взял задачу и выполняет
    - SUCCESS  : выполнена, результат в result.result
    - FAILURE  : упала с ошибкой, исключение в result.result
    - RETRY    : воркер повторяет попытку
    """
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "SUCCESS":
        data = result.result  # dict — то что вернула detect_defects()
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.success,
            defects=data["defects"],
            count=data["count"],
            inference_ms=data["inference_ms"],
        )

    if result.state == "FAILURE":
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.failure,
            error=str(result.result),  # при FAILURE result.result — это исключение
        )

    # PENDING, STARTED, RETRY — задача ещё не готова
    return TaskResponse(task_id=task_id, status=TaskStatus.pending)
