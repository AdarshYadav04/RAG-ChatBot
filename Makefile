.PHONY: install dev test lint format docker-build docker-up docker-down clean

install:
	pip install -r requirements-dev.txt

dev:
	bash scripts/run_dev.sh

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check app/ tests/
	mypy app/

format:
	ruff format app/ tests/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

generate-key:
	python scripts/generate_api_key.py

generate-secret:
	python scripts/generate_api_key.py --type secret-key
