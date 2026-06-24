import time
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from app.models.schemas import BBox, Defect

MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "models" / "best.pt"


class DefectDetector:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nPlace best.pt in ml/models/")
        self.model = YOLO(str(MODEL_PATH), task="detect")
        print(f"✓ Model loaded: {MODEL_PATH.name}")

    def predict(self, image: np.ndarray, confidence: float = 0.25) -> tuple[list[Defect], float]:
        start = time.perf_counter()
        results = self.model.predict(source=image, conf=confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        defects = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            defects.append(Defect(
                class_name=self.model.names[int(box.cls[0])],
                confidence=round(float(box.conf[0]), 4),
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            ))

        return defects, round(elapsed_ms, 2)


_detector: Optional[DefectDetector] = None


def get_detector() -> DefectDetector:
    if _detector is None:
        raise RuntimeError("Detector not initialized")
    return _detector


def load_model() -> None:
    global _detector
    _detector = DefectDetector()


def is_model_loaded() -> bool:
    return _detector is not None
