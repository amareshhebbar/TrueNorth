FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-dev --no-interaction

COPY . .

EXPOSE 8000
CMD ["uvicorn", "truenorth.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
