from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Car Price Prediction API")

class CarFeatures(BaseModel):
    age: int
    mileage: int
    weight: int
    horsepower: int

model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "model/model.joblib")
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # For development, we might not have the model yet
        model = None

@app.post("/predict")
def predict(features: CarFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
