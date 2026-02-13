FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --without dev

COPY . .

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
