# train_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Step 1: Create some dummy data
# Example: Predict y from x where y ≈ 2x + 3
X = np.array([[1], [2], [3], [4], [5]])  # features
y = np.array([5, 7, 9, 11, 13])          # labels

# Step 2: Train a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 3: Test prediction
print("Prediction for x=6:", model.predict([[6]]))

# Step 4: Save the model as pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
