import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("assignment5-ci-cd")

# Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Training
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Log params + metric
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("accuracy", accuracy)

    # Save run id in model_info.txt
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
