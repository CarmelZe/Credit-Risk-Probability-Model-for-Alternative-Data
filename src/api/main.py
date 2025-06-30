from fastapi import FastAPI
import mlflow.pyfunc
from pydantic_models import PredictionInput, PredictionOutput
import pandas as pd

app = FastAPI()

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "risk_probability": float(probability),
        "risk_class": int(prediction)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}