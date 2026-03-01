FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Ставлю минимальный набор системных зависимостей для chroma/pypdf.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Директории для знаний и базы Chroma
RUN mkdir -p /app/books /app/chroma_db

# По умолчанию контейнер запускает Telegram-бота.
CMD ["python", "main.py"]

