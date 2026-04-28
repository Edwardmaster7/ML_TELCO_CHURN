.PHONY: test run

test:
	pytest tests/api/ -v

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
