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
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


@pytest.fixture(scope="module")
def client():
    mock_detector = MagicMock()
    mock_detector.predict.return_value = (
        [Defect(class_name="defect", confidence=0.85, bbox=BBox(x1=10, y1=20, x2=100, y2=200))],
        150.0,
    )

    with patch("app.core.model._detector", mock_detector), \
         patch("app.core.mlflow_tracker.register_deployed_model"):

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            yield

        test_app = FastAPI(lifespan=mock_lifespan)
        test_app.include_router(router)
        test_app.include_router(tasks_router)

        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(test_app).expose(test_app)

        yield TestClient(test_app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_rejects_wrong_content_type(client):
    response = client.post("/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_pdf(client):
    response = client.post("/predict", files={"file": ("doc.pdf", b"%PDF", "application/pdf")})
    assert response.status_code == 415


def test_predict_returns_defects(client, valid_png_bytes):
    response = client.post("/predict", files={"file": ("test.png", valid_png_bytes, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "defects" in data
    assert "count" in data
    assert "inference_ms" in data
    assert data["count"] == len(data["defects"])


def test_predict_response_structure(client, valid_png_bytes):
    response = client.post("/predict", files={"file": ("test.png", valid_png_bytes, "image/png")})
    assert response.status_code == 200
    defect = response.json()["defects"][0]
    assert defect["class_name"] == "defect"
    assert 0.0 <= defect["confidence"] <= 1.0
    assert all(k in defect["bbox"] for k in ("x1", "y1", "x2", "y2"))
