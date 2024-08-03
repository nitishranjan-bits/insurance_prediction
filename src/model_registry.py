import mlflow

def register_best_model(run_id, model_name):
    model_uri = f"runs:/{run_id}/{model_name}"
    mlflow.register_model(model_uri, model_name)
