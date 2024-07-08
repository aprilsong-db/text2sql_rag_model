# Databricks notebook source

import sys
import os

sys.path.append(os.path.abspath('..'))
from text2sql_rag_model.model_deployment.utils import get_latest_model_version, create_or_update_model_endpoint


import mlflow
from mlflow import MlflowClient

from mlflow.deployments import get_deploy_client
from config import CATALOG, SCHEMA, MODEL_NAME, REGISTERED_MODEL_NAME, ENDPOINT_NAME


latest_model_version = get_latest_model_version(REGISTERED_MODEL_NAME)
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
deployment_config = {
    "served_entities": [
        {
            "entity_name": REGISTERED_MODEL_NAME,
            "entity_version": latest_model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
            "environment_vars": {"DATABRICKS_HOST": host, "DATABRICKS_TOKEN": mlflow.utils.databricks_utils.get_databricks_host_creds().token}
        }
    ],
    # "auto_capture_config": {
    #     "catalog_name": CATALOG,
    #     "schema_name": SCHEMA,
    #     "table_name_prefix": f"{MODEL_NAME}",
    # }
}


def main():
    create_or_update_model_endpoint(name=ENDPOINT_NAME, config=deployment_config)

if __name__ == "__main__":
    main()
