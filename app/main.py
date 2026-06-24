from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.api.tasks_routes import router as tasks_router
from app.core.mlflow_tracker import register_deployed_model
from app.core.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    register_deployed_model()
    yield


app = FastAPI(
    title="Visual Inspection Service",
    description="Defect detection API powered by YOLOv8",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(tasks_router)

Instrumentator().instrument(app).expose(app)
