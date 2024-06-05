# Databricks notebook source
from mlflow.tracking import MlflowClient
import mlflow
# from utils import get_latest_model_version



def get_latest_model_version(model_name):
    """Get latest model version, used for POC demo.

    Args:
        model_name (str): Name of ML model

    Returns:
        int: latest version of model for given name
    """
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

UC_MODEL_NAME = "asong_dev.llms.sqlcoder_7b"

def main():
    # Instantiate the client
    client = MlflowClient()

    mlflow.set_registry_uri("databricks-uc")

    # Retrieve the latest model version number
    latest_model_version = get_latest_model_version(UC_MODEL_NAME)
    client.set_registered_model_alias(name=UC_MODEL_NAME, alias="Champion", version=latest_model_version)

if __name__ == "__main__":
    main()

    

# COMMAND ----------


