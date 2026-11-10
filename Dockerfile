FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml ./
COPY src ./src
RUN pip install .
COPY api ./api
COPY app ./app
# data/ y artifacts/ se montan como volúmenes (ver docker-compose.yml)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
