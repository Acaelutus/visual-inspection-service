"""
conftest.py — загружается pytest автоматически перед запуском тестов.

Мокаем тяжёлые пакеты ДО того как pytest импортирует тестовые модули.
Это позволяет запускать тесты без установки ultralytics (PyTorch ~600MB) и mlflow.

sys.modules — словарь всех загруженных модулей Python.
Если поместить MagicMock() вместо реального модуля, любой
`import ultralytics` в app-коде получит MagicMock вместо реального пакета.
"""

import sys
from unittest.mock import MagicMock

# Мокаем ultralytics — иначе app.core.model попытается импортировать PyTorch
sys.modules.setdefault("ultralytics", MagicMock())

# Мокаем mlflow — иначе app.core.mlflow_tracker попытается подключиться к серверу
sys.modules.setdefault("mlflow", MagicMock())
