"""
Регистрация модели в MLflow при запуске сервиса.

MLflow — инструмент для трекинга ML экспериментов и управления моделями.
Здесь мы используем его как Model Registry: фиксируем какая версия модели
задеплоена, когда и с какими параметрами.

В production это даёт ответ на вопрос: "Какая модель сейчас на prod?"
"""

import mlflow

from app.core.config import MLFLOW_TRACKING_URI
from app.core.model import MODEL_PATH


def register_deployed_model() -> None:
    """
    Логирует метаданные текущей модели в MLflow.
    Вызывается один раз при старте сервиса.
    """
    # Не падаем если MLflow недоступен — метрики важнее, но не критичны
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("visual-inspection-deployments")

        with mlflow.start_run(run_name="service-startup"):
            # Параметры модели — строки, числа, пути
            mlflow.log_param("model_path", str(MODEL_PATH))
            mlflow.log_param(
                "model_size_mb",
                round(MODEL_PATH.stat().st_size / 1024 / 1024, 2),
            )
            mlflow.log_param("model_filename", MODEL_PATH.name)

            # Теги — произвольные метки для поиска и фильтрации
            mlflow.set_tag("model_type", "YOLOv8")
            mlflow.set_tag("task", "object_detection")
            mlflow.set_tag("dataset", "MVTec AD")
            mlflow.set_tag("stage", "production")

        print(f"✓ Модель зарегистрирована в MLflow: {MLFLOW_TRACKING_URI}")

    except Exception as e:
        # mlflow недоступен — логируем но не останавливаем сервис
        print(f"⚠ MLflow недоступен, пропускаем регистрацию: {e}")
