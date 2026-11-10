# En Windows sin make, usa los comandos equivalentes del README.
.PHONY: install pipeline test lint api dashboard up down

install:
	pip install -e ".[dev]"

pipeline:
	python -m billboard_intel

test:
	pytest -q

lint:
	ruff check src api app tests

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run app/streamlit_app.py

up:
	docker compose run --rm pipeline && docker compose up --build

down:
	docker compose down
