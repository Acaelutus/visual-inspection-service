"""
Точка входа FastAPI сервиса.

Запуск:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.api.tasks_routes import router as tasks_router
from app.core.mlflow_tracker import register_deployed_model
from app.core.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan — код до yield выполняется при старте, после yield — при остановке.
    """
    print("Загрузка модели...")
    load_model()
    # Регистрируем текущую модель в MLflow как задеплоенную версию
    register_deployed_model()
    yield
    print("Сервис остановлен")


app = FastAPI(
    title="Visual Inspection Service",
    description="Детекция дефектов на производстве с помощью YOLOv8",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(tasks_router)

# Instrumentator автоматически добавляет /metrics endpoint
# и начинает собирать стандартные HTTP метрики: request count, latency, size
Instrumentator().instrument(app).expose(app)
