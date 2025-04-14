# MLflow Tutorial: Wine Quality Prediction

> **Author:** This project and tutorial were created with assistance from Google's AI Assistant, **Gemini**, on April 13, 2025, to help users learn about MLflow through a practical example.

## Overview

This project serves as a step-by-step tutorial for learning how to use [MLflow](https://mlflow.org/) 
to manage the machine learning lifecycle. We use a classic dataset, 
Wine Quality (Red), and progress from a simple training script to a 
more sophisticated setup involving hyperparameter optimization and MLflow tracking.

The goal is to demonstrate core MLflow concepts, including:

* Tracking experiments locally (`mlruns`).
* Logging parameters, metrics, and artifacts (models).
* Using the MLflow UI for comparison.
* Setting up and using a remote MLflow Tracking Server.
* Integrating hyperparameter optimization (`scikit-optimize`) with MLflow tracking.
* Configuring MLflow via code, CLI arguments, and environment variables.

## Tutorial Stages (Reflected in Commit History)

This repository's Git commit history reflects the progression of the tutorial. You can check out earlier commits to see the code at different stages:

1.  **Basic Setup:** Project structure with `Makefile` and `requirements.txt`.
2.  **Initial Training Script:** A simple Python script (`src/train.py`) to load data and train a RandomForestRegressor *without* MLflow.
3.  **Local MLflow Tracking:** Integrated `mlflow.log_param`, `mlflow.log_metric`, `mlflow.log_model` to track runs locally in `mlruns`.
4.  **Configurable Hyperparameters:** Refactored script to accept hyperparameters (`n_estimators`, `max_depth`) via command-line arguments using `argparse` (later refactored to `click`).
5.  **Remote Tracking Server Setup:** Discussed and implemented running `mlflow server` with file backend/artifact stores.
6.  **Hyperparameter Optimization:** Integrated `scikit-optimize` (`gp_minimize`) for Bayesian optimization, logging trials as nested MLflow runs.
7.  **Experiment Naming & Configuration:** Added ways to specify MLflow experiment names and tracking URIs via environment variables and/or CLI arguments.

## Technology Stack

* Python 3
* MLflow
* Scikit-learn
* Pandas
* NumPy
* Click (for CLI arguments)
* Scikit-optimize (for hyper