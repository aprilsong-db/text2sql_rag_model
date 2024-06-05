# Databricks notebook source

import mlflow
from mlflow import MlflowClient
# from utils import get_latest_model_version
from mlflow.deployments import get_deploy_client

from mlflow import MlflowClient

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
    

CATALOG = "asong_dev"
SCHEMA = "llms"
MODEL_NAME = "text2sqlrag"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
MODEL_SERVING_ENDPOINT_NAME = f"{MODEL_NAME}_asong"

latest_model_version = get_latest_model_version(UC_MODEL_NAME)
DEPLOYMENT_CONFIG = {
    "served_entities": [
        {
            "entity_name": f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
            "entity_version": latest_model_version,
            "workload_size": "Small",
            "workload_type": "GPU_MEDIUM",
            "scale_to_zero_enabled": True,
        }
    ],
    "auto_capture_config": {
        "catalog_name": CATALOG,
        "schema_name": SCHEMA,
        "table_name_prefix": f"{MODEL_NAME}",
    }
}

def create_or_update_model_endpoint():
    deploy_client = get_deploy_client("databricks")

    try:
        endpoint = deploy_client.create_endpoint(
            name=MODEL_SERVING_ENDPOINT_NAME,
            config=DEPLOYMENT_CONFIG
        )
    except Exception as e:
        print(f"Endpoint creation failed, attempting update: {e}")
        endpoint = deploy_client.update_endpoint(
            endpoint=MODEL_SERVING_ENDPOINT_NAME,
            config=DEPLOYMENT_CONFIG
        )

def main():

    client = MlflowClient()
    mlflow.set_registry_uri("databricks-uc")

    create_or_update_model_endpoint()


if __name__ == "__main__":
    main()

# COMMAND ----------


