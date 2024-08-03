# insurance_prediction

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

curl -X GET "http://localhost:5000/api/2.0/mlflow/runs/get?run_id=2fee8f75f58f49c7bae21b204ac6f9c0" | jq


Get Model Details:

curl -X GET "http://localhost:5000/api/2.0/mlflow/registered-models/get?name=insurance_prediction_model" | jq


Get Model Version:

curl -X GET "http://localhost:5000/api/2.0/mlflow/model-versions/get?name=insurance_prediction_model&version=1" | jq


Request Body:- 
curl -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d '[
          {"age": 25.0, "bmi": 22.5, "children": 1.0, "sex_female": 0.0, "sex_male": 1.0, "smoker_no": 1.0, "smoker_yes": 0.0, "region_northeast": 0.0, "region_northwest": 1.0, "region_southeast": 0.0, "region_southwest": 0.0},
          {"age": 45.0, "bmi": 28.0, "children": 3.0, "sex_female": 1.0, "sex_male": 0.0, "smoker_no": 0.0, "smoker_yes": 1.0, "region_northeast": 1.0, "region_northwest": 0.0, "region_southeast": 0.0, "region_southwest": 0.0}
        ]' | jq
        
 Output:- 
 
 {
  "predictions": [
    18895.54130247672,
    48346.60243927529
  ]
}




