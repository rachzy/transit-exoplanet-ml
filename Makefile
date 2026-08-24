.PHONY: help install validate validate-train validate-predict evaluate train predict \
        test coverage lint format clean full-pipeline

# Override on the command line, e.g. `make predict TEST_DIR=data/processed/other`.
DATA_DIR ?= data/processed/train
TEST_DIR ?= data/processed/test
ARTIFACT_DIR ?= models
REPORT_DIR ?= reports/nested-cv
OUTPUT ?= predictions.csv
MODEL_DIR ?=

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install/sync dependencies with uv
	uv sync

validate: validate-train validate-predict ## Validate both the train and test data directories

validate-train: ## Validate the training data directory against the train contract
	uv run exoplanet-ml validate --mode train --data-dir $(DATA_DIR)

validate-predict: ## Validate the test data directory against the predict contract
	uv run exoplanet-ml validate --mode predict --data-dir $(TEST_DIR)

evaluate: ## Run nested star-grouped evaluation and write the report
	uv run exoplanet-ml evaluate --data-dir $(DATA_DIR) --output-dir $(REPORT_DIR)

train: ## Train the production stack into a new immutable run directory
	uv run exoplanet-ml train --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR)

predict: ## Score the test data (uses the latest run under ARTIFACT_DIR, or pass MODEL_DIR=path)
	@model_dir="$(MODEL_DIR)"; \
	if [ -z "$$model_dir" ]; then \
		model_dir=$$(ls -dt $(ARTIFACT_DIR)/*/ 2>/dev/null | head -n 1); \
	fi; \
	if [ -z "$$model_dir" ]; then \
		echo "No trained model found under $(ARTIFACT_DIR)/. Run 'make train' first, or pass MODEL_DIR=path/to/run."; \
		exit 1; \
	fi; \
	echo "Using model: $$model_dir"; \
	uv run exoplanet-ml predict --model-dir "$$model_dir" --data-dir $(TEST_DIR) --output $(OUTPUT)

full-pipeline: train predict ## Train the stack, then predict on the test data

test: ## Run the test suite
	uv run pytest

coverage: ## Run the test suite with a coverage report
	uv run coverage run -m pytest
	uv run coverage report

lint: ## Lint the source with ruff
	uv run ruff check src

format: ## Auto-format the source with ruff
	uv run ruff format src

clean: ## Remove generated models, reports, predictions, and caches
	rm -rf $(ARTIFACT_DIR) reports predictions.csv predictions_*.csv
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
