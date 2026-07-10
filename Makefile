.PHONY: install serve dev lint test test-unit test-e2e clean

install:
	uv sync
	cd ui && npm install

serve:
	uv run mmdoc serve

dev:
	cd ui && npm run dev

lint:
	uv run ruff check src tests
	cd ui && npm run lint

test:
	uv run pytest -v

test-unit:
	uv run pytest tests/test_document.py tests/test_vl_model.py tests/test_extractor.py tests/test_api.py -v

test-e2e:
	uv run pytest tests/test_e2e.py -v

clean:
	rm -rf .pytest_cache .ruff_cache
	rm -rf ui/.next ui/node_modules/.cache
