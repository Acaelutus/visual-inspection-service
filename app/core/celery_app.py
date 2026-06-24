from celery import Celery

from app.core.config import REDIS_URL

celery_app = Celery(
    "visual_inspection",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Moscow",
    enable_utc=True,
    result_expires=86400,
)
