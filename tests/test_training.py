import subprocess
import os

def test_model_training_and_artifact_creation():
    """
    Tests if train.py runs successfully and checks if the model file has been created.
    """
    # Run the training script
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    
    # Assertion: The script must run successfully (return code 0)
    assert result.returncode == 0, f"Training script failed: {result.stderr}"
    
    # Assertion: The model file must exist
    assert os.path.exists("artifacts/model.pkl"), "Model file (model.pkl) was not created."
    
    # Assertion: The test data file must exist
    assert os.path.exists("artifacts/test_data.csv"), "Test data file (test_data.csv) was not created."