from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total prediction requests",
    ["endpoint"],
)

DEFECTS_DETECTED_TOTAL = Counter(
    "defects_detected_total",
    "Total defects detected",
)

INFERENCE_DURATION_MS = Histogram(
    "inference_duration_ms",
    "YOLOv8 inference duration in milliseconds",
    buckets=[50, 100, 150, 200, 300, 500, 750, 1000, 2000],
)
