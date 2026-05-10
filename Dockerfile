FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-cache
RUN .venv/bin/python -m spacy download en_core_web_sm

ARG GUARDRAILS_API_KEY=""
RUN if [ -n "$GUARDRAILS_API_KEY" ]; then \
      .venv/bin/guardrails configure --token=$GUARDRAILS_API_KEY --disable-metrics --disable-remote-inferencing && \
      .venv/bin/guardrails hub install hub://guardrails/detect_pii && \
      .venv/bin/guardrails hub install hub://guardrails/toxic_language && \
      .venv/bin/guardrails hub install hub://tryolabs/restricttotopic; \
    else \
      echo "Skipping guardrails hub install (GUARDRAILS_API_KEY not set); custom regex validators will be used at runtime."; \
    fi

COPY . /app

EXPOSE 8000 8501

CMD ["uv", "run", "python", "main.py"]
