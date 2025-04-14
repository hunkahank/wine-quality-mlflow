"""
Training script for Wine Quality prediction using scikit-optimize for
hyperparameter tuning and MLflow for tracking.

Performs Bayesian Optimization using gp_minimize to find optimal hyperparameters
for a RandomForestRegressor, logging each trial as a nested MLflow run.

MLflow Tracking URI and Experiment Name can be set via command-line arguments.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import logging
import sys
import click
import mlflow
import mlflow.sklearn

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
TARGET_COLUMN = "quality"
RANDOM_STATE = 42
OPTIMIZE_METRIC = "rmse"

# --- Data Loading Function ---
def load_data(url: str, separator: str = ';') -> pd.DataFrame:
    """Loads data from a URL into a pandas DataFrame."""
    try:
        logging.info(f"Loading data from {url}...")
        df = pd.read_csv(url, sep=separator)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {url}: {e}")
        sys.exit(1)

# --- Define Hyperparameter Search Space ---
search_space = [
    Integer(50, 500, name='n_estimators'),
    Integer(5, 50, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf'),
]
search_space_map = {dim.name: dim for dim in search_space}

# --- Main CLI Command using Click ---

@click.command()
@click.option(
    "--tracking-uri",
    default=None, # Default to None, MLflow will use local if not set
    type=str,
    help="MLflow Tracking URI (e.g., http://server:port or file:/path/to/mlruns)."
)
@click.option(
    "--experiment-name",
    default="Default", # Default to MLflow's default experiment
    type=str,
    show_default=True,
    help="Name of the MLflow experiment to log runs under."
)
@click.option(
    "--n-calls",
    default=20,
    type=int,
    show_default=True,
    help="Number of hyperparameter optimization calls (trials)."
)
@click.option(
    "--n-initial-points",
    default=10,
    type=int,
    show_default=True,
    help="Number of initial random points before Bayesian optimization."
)
def optimize_train(tracking_uri, experiment_name, n_calls, n_initial_points): # Add tracking_uri parameter
    """Performs hyperparameter optimization using skopt and trains the final best model."""

    # --- Configure MLflow --- # <-- Add this block at the start
    if tracking_uri:
        logging.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        logging.info("Using default MLflow tracking URI (local ./mlruns).")

    if experiment_name and experiment_name != "Default":
        logging.info(f"Setting MLflow experiment to: {experiment_name}")
        mlflow.set_experiment(experiment_name)
    else:
        logging.info("Using default MLflow experiment.")
    # --- End MLflow Configuration ---

    logging.info(f"Starting hyperparameter optimization with {n_calls} calls ({n_initial_points} initial).")

    # --- Load Data, Prepare, Split ---
    # (No changes needed here)
    wine_df = load_data(DATA_URL)
    if TARGET_COLUMN not in wine_df.columns: logging.error(f"Target column '{TARGET_COLUMN}' not found."); sys.exit(1)
    X = wine_df.drop(TARGET_COLUMN, axis=1); y = wine_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    logging.info(f"Data split complete.")


    # --- Objective Function ---
    # (No changes needed here)
    @use_named_args(search_space)
    def objective(**params):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            # ... (rest of objective function: train, evaluate, log metrics) ...
            rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mlflow.log_metric(OPTIMIZE_METRIC, rmse)
            # Optionally log other metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mlflow.log_metric("mae", mae); mlflow.log_metric("r2", r2)
        return rmse

    # --- Start the Main (Parent) MLflow Run ---
    # The run will now go into the experiment and URI set above
    with mlflow.start_run(run_name="Hyperparameter Optimization Parent"):
        mlflow.set_tag("optimization_tool", "scikit-optimize")
        mlflow.set_tag("optimization_goal", f"minimize_{OPTIMIZE_METRIC}")
        mlflow.log_param("n_calls", n_calls)
        mlflow.log_param("n_initial_points", n_initial_points)
        for dim in search_space:
             mlflow.log_param(f"space_{dim.name}_type", type(dim).__name__)
             mlflow.log_param(f"space_{dim.name}_range", f"{dim.low}-{dim.high}")

        logging.info("Starting gp_minimize optimization process...")
        # --- Run Bayesian Optimization ---
        result = gp_minimize(
            func=objective, dimensions=search_space, n_calls=n_calls,
            n_initial_points=n_initial_points, acq_func="EI", random_state=RANDOM_STATE
        )
        logging.info("Optimization process finished.")

        # --- Log Best Results ---
        best_params_list = result.x; best_rmse = result.fun
        best_params_dict = {dim.name: value for dim, value in zip(search_space, best_params_list)}
        logging.info(f"Best parameters found: {best_params_dict}")
        logging.info(f"Best {OPTIMIZE_METRIC} found: {best_rmse:.4f}")
        mlflow.log_metric(f"best_{OPTIMIZE_METRIC}", best_rmse)
        mlflow.log_params({f"best_{name}": value for name, value in best_params_dict.items()})

        # --- Train and Log Final Model ---
        logging.info("Training final model with best parameters...")
        final_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params_dict)
        final_model.fit(X_train, y_train)
        mlflow.sklearn.log_model(final_model, "best-random-forest-model")
        logging.info("Logged final best model.")

    logging.info("Optimization and final model training complete.")


if __name__ == "__main__":
    optimize_train()