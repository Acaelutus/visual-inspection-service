"""
Celery приложение — настройка очереди задач.

Celery нужны два адреса:
- broker: куда ОТПРАВЛЯТЬ задачи (очередь). FastAPI пишет сюда.
- backend: куда СОХРАНЯТЬ результаты. Воркер пишет, клиент читает.

Мы используем Redis для обеих ролей — проще всего для старта.
"""

from celery import Celery

from app.core.config import REDIS_URL

# Celery() — создаём экземпляр приложения
# include — список модулей где искать задачи (импортируется при старте воркера)
celery_app = Celery(
    "visual_inspection",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",    # задачи сериализуем в JSON (не pickle — безопаснее)
    result_serializer="json",  # результаты тоже в JSON
    accept_content=["json"],   # принимаем только JSON
    timezone="Europe/Moscow",
    enable_utc=True,
    result_expires=86400,      # результат хранится в Redis 24 часа, потом удаляется
)
