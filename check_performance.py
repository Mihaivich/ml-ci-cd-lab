import subprocess
import sys

# Define the performance threshold
PERFORMANCE_THRESHOLD = 0.99

# Run the evaluation script and capture its output (i.e., the accuracy)
result = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True)

if result.returncode != 0:
    print(f"Evaluation script failed to execute: {result.stderr}")
    sys.exit(1)

# Convert the output string to a floating-point number
try:
    accuracy = float(result.stdout.strip())
except ValueError:
    print(f"Could not parse accuracy from the evaluation script's output: {result.stdout}")
    sys.exit(1)


print(f"Model Accuracy: {accuracy:.2f}")
print(f"Performance Threshold: {PERFORMANCE_THRESHOLD:.2f}")

# Check if the performance meets the threshold
if accuracy < PERFORMANCE_THRESHOLD:
    print("Model performance did not meet the threshold!")
    sys.exit(1)  # Critical! Exit with a non-zero status code to indicate failure
else:
    print("Model performance meets the threshold.")
    sys.exit(0)  # Exit with a zero status code to indicate success