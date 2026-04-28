.PHONY: test run

test:
	uv run pytest tests/ -v

run:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
