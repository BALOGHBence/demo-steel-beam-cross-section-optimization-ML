import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_ARTIFACT_LOCATION = os.environ.get("MLFLOW_ARTIFACT_LOCATION", "./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def create_experiment(name: str):
    try:
        mlflow.create_experiment(name, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    except mlflow.exceptions.MlflowException:
        pass  # Experiment already exists
    

create_experiment("failure_prediction")
create_experiment("section_estimation")
create_experiment("utilization_estimation")
create_experiment("geometry_validation")