FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system solarpulse \
    && adduser --system --ingroup solarpulse solarpulse

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

USER solarpulse

EXPOSE 8000

CMD ["uvicorn", "solarpulse_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
