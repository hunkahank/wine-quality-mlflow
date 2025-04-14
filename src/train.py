"""
Training script for the Wine Quality prediction model with MLflow tracking
and configurable hyperparameters via command-line arguments using Click.

This script loads the wine quality dataset, trains a RandomForestRegressor model,
evaluates its performance, and logs parameters, metrics, and the model
artifact using MLflow. Hyperparameters can be set via command-line args.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import sys
import click # Import Click
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
TARGET_COLUMN = "quality"
RANDOM_STATE = 42

# --- Functions ---
# (load_data and evaluate_model functions remain the same)
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

def evaluate_model(y_true, y_pred):
    """Calculates and returns evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

# --- Main CLI Command using Click ---

@click.command()
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True, # Show default value in help message
    help="Number of trees in the random forest."
)
@click.option(
    "--max-depth",
    default=10,
    type=int,
    show_default=True,
    help="Maximum depth of the trees. Use 0 for unlimited depth."
)
def train(n_estimators, max_depth):
    """Trains and evaluates a RandomForestRegressor model on the Wine Quality dataset with MLflow tracking."""

    # Use None for max_depth if 0 is passed, as sklearn expects None or int > 0
    parsed_max_depth = max_depth if max_depth > 0 else None

    logging.info(f"Starting training script with args: n_estimators={n_estimators}, max_depth={parsed_max_depth}")

    # --- MLflow Tracking Start ---
    with mlflow.start_run():
        # Log the parameters received from command line *and* constants
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", parsed_max_depth if parsed_max_depth is not None else "None") # Log actual value used
        mlflow.log_param("cli_max_depth_raw", max_depth) # Log raw input arg value too
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("data_url", DATA_URL)
        mlflow.log_param("train_test_split", 0.2)
        logging.info(f"MLflow run started. Logged parameters.")

        # 1. Load Data
        wine_df = load_data(DATA_URL)
        if TARGET_COLUMN not in wine_df.columns:
            logging.error(f"Target column '{TARGET_COLUMN}' not found.")
            mlflow.end_run(status='FAILED') # End run explicitly on error
            sys.exit(1)

        # 2. Prepare Data
        X = wine_df.drop(TARGET_COLUMN, axis=1)
        y = wine_df[TARGET_COLUMN]

        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        logging.info(f"Data split complete.")

        # 4. Initialize and Train Model using click parameters
        logging.info(f"Training RandomForestRegressor...")
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators, # Use parameter directly
            max_depth=parsed_max_depth, # Use potentially modified value
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        logging.info("Model training complete.")

        # 5. Make Predictions
        y_pred = rf_model.predict(X_test)

        # 6. Evaluate Model
        metrics = evaluate_model(y_test, y_pred)
        logging.info("Model evaluation complete.")

        # 7. Log Metrics
        mlflow.log_metrics(metrics) # Use log_metrics for dictionary
        logging.info(f"Logged metrics: {metrics}")

        # 8. Log Model
        mlflow.sklearn.log_model(rf_model, "random-forest-model")
        logging.info("Logged model artifact.")

        logging.info("MLflow run finished successfully.")
    # --- MLflow Tracking End ---

    # Print results (optional)
    print("\n--- Model Evaluation Metrics (also logged in MLflow) ---")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")
    print("--------------------------------------------------------")

    logging.info("Training script finished.")


if __name__ == "__main__":
    train() # Execute the click command function