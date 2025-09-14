# Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1️⃣ Generate a sample dataset
# -----------------------------
X, y = make_classification(
    n_samples=1000,      # number of samples
    n_features=10,       # number of features
    n_informative=6,     # informative features
    n_redundant=2,       # redundant features
    n_classes=2,         # binary classification
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['target'] = y

print("Sample DataFrame:")
print(df.head())

# -----------------------------
# 2️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1),
    df['target'],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 3️⃣ Hyperparameter Tuning
# -----------------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

# -----------------------------
# 4️⃣ Evaluate Model
# -----------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#saving model

import joblib

# Save the trained model
joblib.dump(best_model, "rf_model.pkl")
