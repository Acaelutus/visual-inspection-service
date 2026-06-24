"""
Кастомные Prometheus метрики сервиса.

Prometheus работает с несколькими типами метрик:
- Counter  — только растёт (кол-во запросов, ошибок, дефектов)
- Histogram — распределение значений (время inference, размер ответа)
- Gauge    — может расти и падать (текущее кол-во активных соединений)

Мы используем Counter и Histogram.
"""

from prometheus_client import Counter, Histogram

# Counter — сколько всего предсказаний сделано
# labels=["endpoint"] позволяет фильтровать: sync vs async в Grafana
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Общее количество запросов на детекцию дефектов",
    ["endpoint"],  # метка: "sync" или "async"
)

# Counter — сколько дефектов обнаружено за всё время работы сервиса
DEFECTS_DETECTED_TOTAL = Counter(
    "defects_detected_total",
    "Общее количество обнаруженных дефектов",
)

# Histogram — распределение времени inference модели
# buckets — границы "корзин" гистограммы (в миллисекундах)
# Prometheus сам считает: сколько запросов уложилось в каждую корзину
INFERENCE_DURATION_MS = Histogram(
    "inference_duration_ms",
    "Время inference YOLOv8 в миллисекундах",
    buckets=[50, 100, 150, 200, 300, 500, 750, 1000, 2000],
)
