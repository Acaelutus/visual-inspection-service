import mlflow

from app.core.config import MLFLOW_TRACKING_URI
from app.core.model import MODEL_PATH


def register_deployed_model() -> None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("visual-inspection-deployments")

        with mlflow.start_run(run_name="service-startup"):
            mlflow.log_param("model_path", str(MODEL_PATH))
            mlflow.log_param("model_size_mb", round(MODEL_PATH.stat().st_size / 1024 / 1024, 2))
            mlflow.log_param("model_filename", MODEL_PATH.name)
            mlflow.set_tag("model_type", "YOLOv8")
            mlflow.set_tag("task", "object_detection")
            mlflow.set_tag("dataset", "MVTec AD")
            mlflow.set_tag("stage", "production")

    except Exception:
        # MLflow is optional — service must start even if tracking is unavailable
        pass
