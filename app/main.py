"""
Точка входа FastAPI сервиса.

Запуск:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan — код до yield выполняется при старте, после yield — при остановке.
    Загружаем модель один раз при старте, а не при каждом запросе.
    """
    print("Загрузка модели...")
    load_model()
    yield
    # Место для cleanup при остановке (освобождение GPU памяти и т.д.)
    print("Сервис остановлен")


app = FastAPI(
    title="Visual Inspection Service",
    description="Детекция дефектов на производстве с помощью YOLOv8",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
