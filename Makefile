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

# Lint code
lint: ## Lint code
	flake8 *.py
	pylint *.py

# Clean up
clean: ## Clean up temporary and build files
	rm -rf __pycache__ .pytest_cache *.egg-info dist build
