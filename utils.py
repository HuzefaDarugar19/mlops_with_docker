import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

MODEL_PATH = "model.pkl"

def train_model():
    """
    Generate sample data and train a linear regression model.
    """
    # Generate sample data
    X = np.linspace(0, 10, 100)
    y = 2 * X + 1 + np.random.randn(100) * 0.5  # Add some noise
    X = X.reshape(-1, 1)  # Reshape for sklearn

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving the model: {e}")

if __name__ == "__main__":
    train_model()