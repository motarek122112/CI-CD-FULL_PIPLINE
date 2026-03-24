import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

THRESHOLD = 0.85

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = client.get_run(run_id)
metrics = run.data.metrics

accuracy = metrics.get("accuracy")

if accuracy is None:
    print("Error: accuracy metric not found in MLflow run.")
    sys.exit(1)

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print("Model accuracy is below threshold. Failing pipeline.")
    sys.exit(1)

print("Model passed threshold. Deployment can continue.")
