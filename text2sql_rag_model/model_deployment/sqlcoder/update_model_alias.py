# Databricks notebook source
from mlflow.tracking import MlflowClient
import mlflow
from utils import get_latest_model_version

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

from utils import get_latest_model_version


# COMMAND ----------


