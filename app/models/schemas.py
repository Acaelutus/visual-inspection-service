"""
Pydantic схемы — контракт API.
Определяют что принимает и что возвращает каждый endpoint.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Координаты bounding box в пикселях."""
    x1: int
    y1: int
    x2: int
    y2: int


class Defect(BaseModel):
    """Один найденный дефект."""
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)  # ge=greater or equal, le=less or equal
    bbox: BBox


class PredictResponse(BaseModel):
    """Ответ на запрос детекции."""
    defects: list[Defect]
    count: int
    inference_ms: float  # время inference в миллисекундах


class HealthResponse(BaseModel):
    """Ответ health check — мониторинг использует этот endpoint."""
    status: str
    model_loaded: bool


class TaskStatus(str, Enum):
    """
    Статус задачи Celery.
    str + Enum — FastAPI сериализует как строку ("pending"), не число.
    """
    pending = "pending"
    success = "success"
    failure = "failure"


class TaskResponse(BaseModel):
    """
    Ответ на запросы /tasks/predict и /tasks/{task_id}.
    Поля defects/count/inference_ms заполнены только при status=success.
    Поле error заполнено только при status=failure.
    """
    task_id: str
    status: TaskStatus
    defects: Optional[list[Defect]] = None
    count: Optional[int] = None
    inference_ms: Optional[float] = None
    error: Optional[str] = None
