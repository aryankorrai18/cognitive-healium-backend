FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY healium/ healium/
COPY server.py .
COPY static/ static/

RUN pip install --no-cache-dir ".[redis,postgres]"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
