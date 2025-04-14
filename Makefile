# ----- Variables -----

# Use python3 by default
PYTHON = python3
# Virtual environment directory name
VENV_DIR = /home/ubuntu/.virtualenvs/wine-quality-mlflow
# Activate script (adjust if using fish, etc.)
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
# Python executable within the virtual environment
VENV_PYTHON = $(VENV_DIR)/bin/python
# Pip executable within the virtual environment
VENV_PIP = $(VENV_DIR)/bin/pip
# Requirements file
REQS = requirements.txt
# SRC directory for source
SRC_DIR = src
# Main training script
TRAIN_SCRIPT = $(SRC_DIR)/train.py

# ----- Default Run Configuration -----
# Values used by the 'make run' target if not overridden
DEFAULT_EXPERIMENT_NAME = Wine Opt Skopt CLI Config
DEFAULT_N_CALLS = 30
DEFAULT_N_INITIAL_POINTS = 10

# Default target
.DEFAULT_GOAL := help

# Help text
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Install dependencies using pip from the virtual environment
install: ## Install requirements
	@echo ">>> Installing dependencies from $(REQS)..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r $(REQS)
	@echo ">>> Installation complete."

# Run the main training script using the virtual environment's python
# Ensure environment exists, assumes dependencies are installed via 'make install' or manually
run: ## Run the main training script
	@echo ">>> Running training/optimization script: $(TRAIN_SCRIPT)..."
	@echo ">>> Using Experiment: '$(DEFAULT_EXPERIMENT_NAME)', Calls: $(DEFAULT_N_CALLS), Initial Points: $(DEFAULT_N_INITIAL_POINTS)"
	@if [ -z "$(MLFLOW_TRACKING_URI)" ]; then \
		echo ">>> MLFLOW_TRACKING_URI not set, using local tracking."; \
	else \
		echo ">>> Using MLFLOW_TRACKING_URI: $(MLFLOW_TRACKING_URI)"; \
	fi
	$(VENV_PYTHON) $(TRAIN_SCRIPT) \
		--tracking-uri "$(MLFLOW_TRACKING_URI)" \
		--experiment-name "$(DEFAULT_EXPERIMENT_NAME)" \
		--n-calls $(DEFAULT_N_CALLS) \
		--n-initial-points $(DEFAULT_N_INITIAL_POINTS)
	@echo ">>> Script finished."
	@echo ">>> Training script finished."

# Lint code
lint: ## Lint code
	flake8 *.py
	pylint *.py

# Clean up
clean: ## Clean up temporary and build files
	rm -rf __pycache__ .pytest_cache *.egg-info dist build
