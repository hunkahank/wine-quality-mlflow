"""
Training script for the Wine Quality prediction model with MLflow tracking.

This script loads the wine quality dataset, trains a RandomForestRegressor model,
evaluates its performance, and logs parameters, metrics, and the model
artifact using MLflow for local tracking in the 'mlruns' directory.

-----------------------------
Code Generation Reference:
-----------------------------
This script was generated with assistance from Google's AI Assistant (Gemini).
Date: April 13, 2025

License Information:
The code is provided "as-is" without any warranty, express or implied.
Users are responsible for ensuring that their use of this code complies with
all applicable laws, regulations, and the terms of service of Google's AI tools.

Users should also ensure compliance with the licenses of any libraries imported
or used by this script (e.g., pandas, scikit-learn, numpy, mlflow).
These libraries typically have their own open-source licenses (like BSD, MIT, Apache)
that govern their use, redistribution, and modification.

Responsibility for the application, redistribution, or creation of derivative
works based on this code lies solely with the user. It is recommended to review
Google's AI terms and the specific licenses of all dependencies.
-----------------------------
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import sys
import mlflow # Import MLflow
import mlflow.sklearn # Import the sklearn flavor for model logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
TARGET_COLUMN = "quality"
RANDOM_STATE = 42

# --- Functions ---

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

# --- Main Execution ---

if __name__ == "__main__":
    logging.info("Starting training script with MLflow tracking...")

    # 1. Load Data
    wine_df = load_data(DATA_URL)

    if TARGET_COLUMN not in wine_df.columns:
        logging.error(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
        sys.exit(1)

    # 2. Prepare Data
    X = wine_df.drop(TARGET_COLUMN, axis=1)
    y = wine_df[TARGET_COLUMN]

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logging.info(f"Data split into training ({X_train.shape[0]} samples) and test sets ({X_test.shape[0]} samples).")

    # Define hyperparameters BEFORE starting the MLflow run
    n_estimators = 100
    max_depth = 10

    # --- MLflow Tracking Start ---
    # Start an MLflow run. The 'with' statement ensures the run is ended
    # automatically, even if errors occur.
    with mlflow.start_run():
        logging.info("MLflow run started.")

        # 4. Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("data_url", DATA_URL) # Log data source
        mlflow.log_param("train_test_split", 0.2) # Log test split size
        logging.info(f"Logged parameters: n_estimators={n_estimators}, max_depth={max_depth}, etc.")

        # 5. Initialize and Train Model
        logging.info(f"Training RandomForestRegressor...")
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        logging.info("Model training complete.")

        # 6. Make Predictions
        y_pred = rf_model.predict(X_test)

        # 7. Evaluate Model
        metrics = evaluate_model(y_test, y_pred)
        logging.info("Model evaluation complete.")

        # 8. Log Metrics
        mlflow.log_metric("rmse", metrics['rmse'])
        mlflow.log_metric("mae", metrics['mae'])
        mlflow.log_metric("r2", metrics['r2'])
        # Alternative: log all metrics from the dictionary at once
        # mlflow.log_metrics(metrics)
        logging.info(f"Logged metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

        # 9. Log Model
        # Use the scikit-learn flavor to log the model
        mlflow.sklearn.log_model(rf_model, "random-forest-model")
        logging.info("Logged model artifact 'random-forest-model'.")

        # The 'with' block automatically ends the MLflow run here
        logging.info("MLflow run finished.")

    # --- MLflow Tracking End ---


    # Print results (optional, as they are now logged in MLflow)
    print("\n--- Model Evaluation Metrics (also logged in MLflow) ---")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")
    print("--------------------------------------------------------")

    logging.info("Training script finished successfully.")