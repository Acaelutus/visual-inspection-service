from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Defect(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox


class PredictResponse(BaseModel):
    defects: list[Defect]
    count: int
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class TaskStatus(str, Enum):
    pending = "pending"
    success = "success"
    failure = "failure"


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    defects: Optional[list[Defect]] = None
    count: Optional[int] = None
    inference_ms: Optional[float] = None
    error: Optional[str] = None
