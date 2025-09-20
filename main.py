from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import predict
import uvicorn

app = FastAPI()

class PredictionRequest(BaseModel):
    input_feature: float

class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    try:
        prediction = predict(request.input_feature)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)