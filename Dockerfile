FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "if [ \"$SERVICE_ROLE\" = \"worker\" ]; then python -m backend.worker; else python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}; fi"]
