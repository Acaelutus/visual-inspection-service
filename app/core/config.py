"""
Настройки сервиса из переменных окружения.

os.getenv("KEY", default) — читает переменную окружения KEY,
если не задана — возвращает default.
Это стандартный подход: в production задаём через env, локально — дефолты.
"""

import os

# Redis URL — для локальной разработки localhost, в Docker — имя сервиса redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# MLflow Tracking Server URL
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
