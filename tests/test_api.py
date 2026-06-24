"""
Unit тесты для API endpoints.

Модель и внешние сервисы (Redis, MLflow) замокированы —
тесты не требуют best.pt, запущенного Redis или MLflow.

Мокирование (Mock) — подмена реального объекта на фиктивный.
Это позволяет тестировать бизнес-логику изолированно.
"""

import pytest
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.api.routes import router
from app.api.tasks_routes import router as tasks_router
from app.models.schemas import BBox, Defect


@pytest.fixture(scope="module")
def valid_png_bytes() -> bytes:
    """Генерирует валидный PNG 100x100 через OpenCV для использования в тестах."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # серый квадрат
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


@pytest.fixture(scope="module")
def client():
    """
    Тестовый клиент FastAPI с замоканными зависимостями.

    scope="module" — клиент создаётся один раз на весь модуль тестов,
    не пересоздаётся для каждого теста (быстрее).
    """
    # Создаём мок-детектор который возвращает предсказуемый результат
    mock_detector = MagicMock()
    mock_detector.predict.return_value = (
        [Defect(class_name="defect", confidence=0.85, bbox=BBox(x1=10, y1=20, x2=100, y2=200))],
        150.0,  # inference_ms
    )

    # Патчим model module — подменяем глобальный _detector
    with patch("app.core.model._detector", mock_detector), \
         patch("app.core.mlflow_tracker.register_deployed_model"):

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            # Пустой lifespan — не загружаем модель, не ходим в MLflow
            yield

        test_app = FastAPI(lifespan=mock_lifespan)
        test_app.include_router(router)
        test_app.include_router(tasks_router)

        # Prometheus instrumentator тоже регистрируем
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(test_app).expose(test_app)

        yield TestClient(test_app)


# ---- Health check ----

def test_health_returns_ok(client):
    """GET /health возвращает status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


# ---- Валидация входных данных ----

def test_predict_rejects_wrong_content_type(client):
    """POST /predict с текстовым файлом возвращает 415 Unsupported Media Type."""
    response = client.post(
        "/predict",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_pdf(client):
    """POST /predict с PDF файлом возвращает 415."""
    response = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
    )
    assert response.status_code == 415


# ---- Успешный запрос ----

def test_predict_returns_defects(client, valid_png_bytes):
    """POST /predict с валидным PNG возвращает список дефектов."""
    response = client.post(
        "/predict",
        files={"file": ("test.png", valid_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "defects" in data
    assert "count" in data
    assert "inference_ms" in data
    assert data["count"] == len(data["defects"])


def test_predict_response_structure(client, valid_png_bytes):
    """Ответ /predict содержит правильную структуру bbox."""
    response = client.post(
        "/predict",
        files={"file": ("test.png", valid_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    defect = response.json()["defects"][0]
    assert defect["class_name"] == "defect"
    assert 0.0 <= defect["confidence"] <= 1.0
    assert all(k in defect["bbox"] for k in ("x1", "y1", "x2", "y2"))
