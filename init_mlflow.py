import os
from dotenv import load_dotenv
import mlflow
from mlflow.entities import Experiment
from cso.tasks import Task
from cso.logger import get_logger

load_dotenv()

logger = get_logger()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_ARTIFACT_LOCATION = os.environ.get("MLFLOW_ARTIFACT_LOCATION", "./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")


def create_experiment(name: str):
    try:
        mlflow.create_experiment(name, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    except mlflow.exceptions.MlflowException:
        experiments: list[Experiment] = mlflow.search_experiments(filter_string=f"name = '{name}'")
        if len(experiments) > 0:
            logger.info(f"Experiment '{name}' already exists.")
        else:
            logger.error(f"Failed to create or find experiment '{name}'.")
        
        
for task in Task:
    create_experiment(task.value)
    logger.info(f"Created experiment for task: {task.value}")