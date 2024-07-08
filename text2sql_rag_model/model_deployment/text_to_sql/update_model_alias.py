# Databricks notebook source
import sys
import os
from mlflow.tracking import MlflowClient
import mlflow
# from utils import get_latest_model_version

sys.path.append(os.path.abspath('..'))

from text2sql_rag_model.model_deployment.utils import get_latest_model_version
from config import REGISTERED_MODEL_NAME

def main():
    # Instantiate the client
    client = MlflowClient()

    mlflow.set_registry_uri("databricks-uc")

    # Retrieve the latest model version number
    latest_model_version = get_latest_model_version(REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(name=REGISTERED_MODEL_NAME, alias="Champion", version=latest_model_version)

if __name__ == "__main__":
    main()

    

# COMMAND ----------


