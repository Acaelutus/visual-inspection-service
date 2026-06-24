# Многоэтапная сборка (multi-stage build) — уменьшает размер финального образа.
#
# Идея: на stage builder устанавливаем зависимости (много мусора от pip).
# В финальный образ копируем только установленные пакеты, без build-артефактов.

# ---- Stage 1: Builder — устанавливаем зависимости ----
FROM python:3.9-slim AS builder

WORKDIR /build

# Копируем только requirements — Docker кэширует этот слой.
# Если requirements.txt не изменился, pip install не запускается повторно.
COPY requirements.txt .

# --user: ставим в ~/.local (изолированно от системного Python)
# --no-cache-dir: не кэшируем скачанные пакеты (меньше размер образа)
RUN pip install --no-cache-dir --user -r requirements.txt


# ---- Stage 2: Production — минимальный рабочий образ ----
FROM python:3.9-slim

# libgl1-mesa-glx нужен opencv-python (полная версия) на Linux.
# ultralytics обращается к cv2.imshow при импорте — нужна полная, не headless версия.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем установленные Python пакеты из builder stage
COPY --from=builder /root/.local /root/.local

# Копируем только код приложения (данные и модели монтируются через volume)
COPY app/ ./app/

# Директория для модели — файл best.pt монтируется через volume или скачивается при старте
RUN mkdir -p ml/models

# ~/.local/bin нужен в PATH чтобы uvicorn и celery были доступны как команды
ENV PATH=/root/.local/bin:$PATH

# Документируем что сервис слушает порт 8000
EXPOSE 8000

# Команда по умолчанию — запуск FastAPI.
# Для Celery worker — переопределяем CMD в docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
