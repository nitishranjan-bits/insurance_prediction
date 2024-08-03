import json
import os
import sys
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import preprocess_data
from src.feature_store import save_features, load_features
from src.hyperparameter_tuning import optimize_hyperparameters
from src.model_training import train_and_evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_name = "insurance_prediction"
mlflow.set_experiment(experiment_name)

logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

def save_schema(X):
    schema_file = "schema.json"
    schema = X.dtypes.apply(lambda x: str(x)).to_dict()
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    return schema_file

def run_pipeline():
    try:
        data = pd.read_csv("../data/insurance.csv")
        X, y, feature_names = preprocess_data(data)
        save_features(X, y, feature_names)
        X, y = load_features()

        models = {
            "LinearRegression": {
                "model": LinearRegression(),
                "param_grid": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                }
            },
            "RandomForest": {
                "model": RandomForestRegressor(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "param_grid": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 4, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        }

        best_model = None
        best_score = float('inf')
        best_run_id = None

        for name, model_info in models.items():
            logger.info(f"Training {name} model")
            with mlflow.start_run(run_name=name) as run:
                run_id = run.info.run_id
                logger.info(f"Started run with ID: {run_id}")
                try:
                    best_params = optimize_hyperparameters(model_info["model"], model_info["param_grid"], X, y)
                    model, mse, r2 = train_and_evaluate_model(model_info["model"], best_params, X, y)

                    mlflow.log_params(best_params)
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)

                    signature = infer_signature(X, y)
                    mlflow.sklearn.log_model(model, "model", signature=signature)  # Save model under 'model'

                    schema_file = save_schema(X)
                    mlflow.log_artifact(schema_file)

                    if mse < best_score:
                        best_model = model
                        best_score = mse
                        best_run_id = run.info.run_id

                    logger.info(f"Successfully trained {name} model. MSE: {mse}, R2: {r2}")
                except Exception as e:
                    logger.error(f"Run for model {name} failed due to error: {str(e)}")
                    continue

        if best_run_id:
            logger.info(f"Best model run ID: {best_run_id}")
            model_uri = f"runs:/{best_run_id}/model"
            model_name = "insurance_prediction_model"
            model_description = "This is the best model based on the lowest MSE from the training runs."

            mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered best model from run {best_run_id}")

            client = mlflow.MlflowClient()
            client.set_registered_model_tag(model_name, "production_status", "production")
            client.set_registered_model_tag(model_name, "description", model_description)
            client.set_registered_model_tag(model_name, "schema", "See artifacts for schema details.")

            logger.info(
                f"Tagged model '{model_name}' with 'production_status=production', description, and schema information.")
        else:
            logger.warning("No successful runs to register.")

    except Exception as e:
        logger.error(f"Pipeline failed due to error: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting insurance prediction pipeline")
    run_pipeline()
    logger.info("Pipeline completed")
