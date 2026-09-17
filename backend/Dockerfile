FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY backend ./backend
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY rag ./rag
COPY evaluation ./evaluation
COPY sample_corpus ./sample_corpus
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

