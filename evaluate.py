import pandas as pd
import pickle

# Load the model
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the test data
test_data = pd.read_csv("artifacts/test_data.csv")
X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# Calculate accuracy
accuracy = model.score(X_test, y_test)

# Print the accuracy. This is key! The CI/CD tool will read this value.
print(f"{accuracy}")