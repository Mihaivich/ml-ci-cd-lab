import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle
import os

# Create a directory to store the model
os.makedirs("artifacts", exist_ok=True)

# Load the data
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==> Add code to intentionally create a poorly performing model <==
import numpy as np
print("Intentionally shuffling training labels to test the CI failure scenario...")
np.random.shuffle(y_train)

# Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model to the artifacts folder
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Also save the test set for the evaluation script to use
test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data['target'] = y_test
test_data.to_csv("artifacts/test_data.csv", index=False)

print("Model training complete and saved.")
print("Test data saved.")