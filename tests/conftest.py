import sys
from unittest.mock import MagicMock

sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("mlflow", MagicMock())
