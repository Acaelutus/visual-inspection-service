# Visual Inspection Service

Production-ready defect detection API for manufacturing lines. Built on YOLOv8, trained on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (15 product categories, 1258 labeled images).

## What it does

Accepts an image, returns detected defects with bounding boxes and confidence scores. Designed to replace manual visual inspection on conveyor lines.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@bottle.png" \
  -F "confidence=0.25"
```

```json
{
  "defects": [
    {
      "class_name": "defect",
      "confidence": 0.82,
      "bbox": { "x1": 298, "y1": 264, "x2": 794, "y2": 831 }
    }
  ],
  "count": 1,
  "inference_ms": 269.4
}
```

## Stack

| Layer | Tech |
|---|---|
| Model | YOLOv8 (Ultralytics) |
| API | FastAPI + Uvicorn |
| Async queue | Celery + Redis |
| Observability | Prometheus + Grafana + MLflow |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions → AWS ECR → ECS Fargate |

## Architecture

```
Client
  │
  ├─ POST /predict          → sync inference (~270ms)
  │
  └─ POST /tasks/predict    → enqueue → task_id (immediate)
       GET /tasks/{id}      → poll for result

FastAPI → Celery Worker → YOLOv8 → Redis (results)
                ↓
         Prometheus /metrics → Grafana dashboards
         MLflow tracking → model registry
```

## Quick start

**Prerequisites:** Docker Desktop, Python 3.9, trained `best.pt` in `ml/models/`

```bash
# 1. Clone and set up environment
git clone https://github.com/Acaelutus/visual-inspection-service.git
cd visual-inspection-service
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Start infrastructure
docker compose -f docker/docker-compose.redis.yml up -d
docker compose -f docker/docker-compose.monitoring.yml up -d

# 3. Start API
uvicorn app.main:app --reload --port 8000

# 4. Start worker (separate terminal)
celery -A app.core.celery_app:celery_app worker --loglevel=info --pool=solo
```

Or run the full stack in Docker:

```bash
docker compose up --build
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health + model status |
| `POST` | `/predict` | Synchronous defect detection |
| `POST` | `/tasks/predict` | Async — returns `task_id` immediately |
| `GET` | `/tasks/{task_id}` | Poll task result |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

## Monitoring

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5000 |

## Training

Model was trained on Kaggle (Tesla T4 GPU) using MVTec Anomaly Detection dataset. Binary classification: defect vs. no defect across 15 product categories.

```bash
# Local training
python ml/train/train.py

# Kaggle training (GPU)
# Upload ml/train/kaggle_train.py and data/mvtec_yolo.zip to Kaggle notebook
```

## CI/CD

GitHub Actions pipeline on every push to `master`:

```
lint (ruff) → tests (pytest) → docker build → push to AWS ECR
```

AWS deployment requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in repository secrets.

## Project structure

```
app/
├── api/          # FastAPI routers
├── core/         # model, celery, metrics, config
├── models/       # Pydantic schemas
└── workers/      # Celery tasks

ml/
├── train/        # training and dataset preparation scripts
└── models/       # trained weights (not tracked in git)

docker/
├── prometheus/   # scrape config
└── grafana/      # provisioned dashboards
```
