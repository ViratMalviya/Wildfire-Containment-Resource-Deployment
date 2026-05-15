# ──────────────────────────────────────────────
# Makefile — Wildfire RL + MLOps Automation
# ──────────────────────────────────────────────
# Declarative automation for reproducible experiments.
#
# Usage:
#   make install        # Install dependencies
#   make train          # Train both experiments
#   make evaluate       # Evaluate both experiments
#   make all            # Full pipeline
#   make mlflow         # Start MLflow UI
#   make docker-up      # Start Docker Compose stack
#   make clean          # Remove generated files
# ──────────────────────────────────────────────

.PHONY: install train evaluate all mlflow docker-up docker-down clean help

# ── Variables ──
PYTHON     = python
PIP        = pip
CONFIG_V1  = configs/qlearning_v1.yaml
CONFIG_V2  = configs/qlearning_v2.yaml

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train: ## Train both experiments (v1 + v2)
	$(PYTHON) train.py --config $(CONFIG_V1)
	$(PYTHON) train.py --config $(CONFIG_V2)

evaluate: ## Evaluate both experiments
	$(PYTHON) evaluate.py --config $(CONFIG_V1)
	$(PYTHON) evaluate.py --config $(CONFIG_V2)

all: install train evaluate ## Full pipeline: install → train → evaluate

mlflow: ## Start MLflow tracking UI on port 5000
	mlflow ui --port 5000

docker-up: ## Start full Docker Compose stack (MLflow + Train + Evaluate)
	docker-compose up --build

docker-down: ## Stop Docker Compose stack
	docker-compose down -v

clean: ## Remove generated results, models, and mlruns
	rm -rf results/*.csv results/*.json results/*.png
	rm -rf models/*.pkl
	rm -rf mlruns/
	rm -rf __pycache__ src/__pycache__ sim/__pycache__

tag-experiments: ## Tag current experiments in git
	git tag -f exp-qlearning-1
	git tag -f exp-qlearning-2
	@echo "✅ Experiment tags created"

lint: ## Run flake8 linting
	flake8 --max-line-length=120 --ignore=E501,W503 train.py evaluate.py src/ sim/
