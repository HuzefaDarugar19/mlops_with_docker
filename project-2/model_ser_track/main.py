# main.py
from fastapi import FastAPI
import pickle
import numpy as np
import mlflow
import mlflow.sklearn

<<<<<<< HEAD
mlflow.set_tracking_uri("http://mlflow:5000")  


=======
>>>>>>> 74564bc6d69fecb53cef660e16459815ac75ba56
# Step 1: Create the FastAPI app
app = FastAPI(title="ML Model API with MLflow")

# Step 2: Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Step 3: Log model to MLflow (only once at startup)
mlflow.set_experiment("Simple-Linear-Regression")

with mlflow.start_run(run_name="fastapi-serve", nested=True):
    mlflow.sklearn.log_model(model, name="linear_regression_model")
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("framework", "scikit-learn")

# Step 4: Define routes
@app.get("/")
def home():
    return {
        "message": "Welcome to the ML Model API with MLflow. "
                   "Use /predict?x=number to get predictions."
    }

@app.get("/predict")
def predict(x: float):
    prediction = model.predict(np.array([[x]]))
    
    # Log the prediction to MLflow
    with mlflow.start_run(run_name="prediction-requests", nested=True):
        mlflow.log_param("input_x", x)
        mlflow.log_metric("prediction", float(prediction[0]))
    
    return {"input": x, "prediction": float(prediction[0])}
