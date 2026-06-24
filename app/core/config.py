import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
