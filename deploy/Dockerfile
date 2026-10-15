FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY demo ./demo
RUN mkdir -p /app/data/cco \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8092

CMD ["python", "-m", "uvicorn", "demo.patient_chat.app:app", "--host", "0.0.0.0", "--port", "8092", "--no-access-log"]
