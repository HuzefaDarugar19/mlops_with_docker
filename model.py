import pickle

MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
except Exception as e:
    raise Exception(f"Error loading the model: {e}")

def predict(input_feature):
    """
    Make a prediction using the loaded linear regression model.
    """
    try:
        prediction = model.predict([[input_feature]])[0]
        return prediction
    except Exception as e:
        raise Exception(f"Prediction error: {e}")