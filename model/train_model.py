import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# Set MLflow tracking URI (optional, defaults to ./mlruns)
# mlflow.set_tracking_uri("http://localhost:5000") 

def generate_data(n_samples=1000):
    np.random.seed(42)
    age = np.random.randint(1, 20, n_samples)
    mileage = np.random.randint(5000, 200000, n_samples)
    weight = np.random.randint(2000, 5000, n_samples)
    horsepower = np.random.randint(100, 400, n_samples)
    
    # Synthetic price formula
    price = (
        30000 
        - (age * 1000) 
        - (mileage * 0.05) 
        + (weight * 2) 
        + (horsepower * 50) 
        + np.random.normal(0, 2000, n_samples)
    )
    price = np.maximum(price, 500) # Ensure price is positive
    
    df = pd.DataFrame({
        'age': age,
        'mileage': mileage,
        'weight': weight,
        'horsepower': horsepower,
        'price': price
    })
    return df

def train():
    print("Generating data...")
    df = generate_data()
    
    X = df[['age', 'mileage', 'weight', 'horsepower']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Starting MLflow run...")
    mlflow.set_experiment("car_price_prediction")
    with mlflow.start_run():
        n_estimators = 100
        max_depth = 10
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        print("Model saved to MLflow.")
        
        # Save locally for easy access by API
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved locally to {model_path}")

if __name__ == "__main__":
    train()
