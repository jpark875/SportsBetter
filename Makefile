# ============================================================
# NBA Bet Portfolio Engine — Makefile
# ============================================================
# Common commands. Run `make help` for a summary.

PYTHON      := python
VENV        := .venv
PIP         := $(VENV)/Scripts/pip        # Windows
ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV)/Scripts/activate
else
    PIP      := $(VENV)/bin/pip
    ACTIVATE := $(VENV)/bin/activate
endif

.PHONY: help install env test coverage benchmark train run clean

# ---- Meta ----

help:
	@echo ""
	@echo "  make env         Create virtual environment and install dependencies"
	@echo "  make install     Install dependencies into active environment"
	@echo "  make test        Run unit tests"
	@echo "  make coverage    Run tests with coverage report"
	@echo "  make benchmark   Run ML efficiency benchmark (no API keys required)"
	@echo "  make train       Retrain model on cached data and save artefacts"
	@echo "  make run         Launch the engine interactively"
	@echo "  make clean       Remove __pycache__, .pytest_cache, artefacts"
	@echo ""

# ---- Setup ----

env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\nEnvironment ready. Activate with: source $(ACTIVATE)"

install:
	pip install -r requirements.txt

# ---- Testing ----

test:
	pytest tests/ -v --tb=short

coverage:
	pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
	@echo "\nHTML report → htmlcov/index.html"

# ---- Benchmark (self-contained, no API keys) ----

benchmark:
	$(PYTHON) scripts/benchmark_model.py

# ---- Model training (requires .env with API keys) ----

train:
	$(PYTHON) main.py --train

# ---- Run ----

run:
	$(PYTHON) main.py

# ---- Cleanup ----

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".coverage" -delete 2>/dev/null; true
	@echo "Clean."
