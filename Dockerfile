FROM python:3.11-slim

# XGBoost needs OpenMP at runtime; curl is for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ANALYTICS_DB=/data/incidents.db

COPY analytics/requirements.txt analytics/requirements.txt
RUN pip install --no-cache-dir -r analytics/requirements.txt

COPY shared/ shared/
COPY data/ data/
COPY analytics/ analytics/

VOLUME ["/data"]
EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs http://localhost:8002/api/analytics/health || exit 1

# Keep ONE worker: models live in memory and the hash-chained log needs a single writer
CMD ["uvicorn", "analytics.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]