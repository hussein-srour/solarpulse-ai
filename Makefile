.PHONY: install format lint type-check test check run docker-up docker-down

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

format:
	ruff check --fix .
	ruff format .

lint:
	ruff check .
	ruff format --check .

type-check:
	mypy src tests

test:
	pytest

check: lint type-check test

run:
	uvicorn solarpulse_ai.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down
