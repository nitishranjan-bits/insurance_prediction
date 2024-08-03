import os
import mlflow
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_name = "insurance_prediction_model"
model = None

def load_model():
    global model
    try:
        # Use "latest" to always load the latest version of the model
        model_uri = f"models:/{model_name}/latest"
        print(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        input_data = pd.DataFrame(data)
        predictions = model.predict(input_data)
        return jsonify(predictions=predictions.tolist())
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=True)
