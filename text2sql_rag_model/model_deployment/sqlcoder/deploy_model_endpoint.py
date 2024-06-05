# Databricks notebook source

import mlflow
from mlflow import MlflowClient
from utils import get_latest_model_version
from mlflow.deployments import get_deploy_client

CATALOG = "asong_dev"
SCHEMA = "llms"
MODEL_NAME = "sqlcoder_7b"
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


