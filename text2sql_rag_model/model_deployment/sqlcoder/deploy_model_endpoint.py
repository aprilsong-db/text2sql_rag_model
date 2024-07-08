# Databricks notebook source
import sys
import os
import requests
import json
import mlflow

sys.path.append(os.path.abspath('..'))
from text2sql_rag_model.model_deployment.utils import get_latest_model_version, get_max_provisioned_throughput, create_or_update_model_endpoint
from config import CATALOG, SCHEMA, MODEL_NAME, REGISTERED_MODEL_NAME, ENDPOINT_NAME


def main():

    client = mlflow.MlflowClient()
    mlflow.set_registry_uri("databricks-uc")
    latest_model_version = get_latest_model_version(REGISTERED_MODEL_NAME)
    max_provisioned_throughput = get_max_provisioned_throughput(REGISTERED_MODEL_NAME, latest_model_version)

    deployment_config = {
        "served_entities": [
            {
                "entity_name": REGISTERED_MODEL_NAME,
                "entity_version": latest_model_version,
                "max_provisioned_throughput": max_provisioned_throughput,
                "scale_to_zero_enabled": True,
                "workload_size": "Small"
            }
        ],
        "auto_capture_config": {
            "catalog_name": CATALOG,
            "schema_name": SCHEMA,
            "table_name_prefix": ENDPOINT_NAME,
        }
    }

    create_or_update_model_endpoint(name=ENDPOINT_NAME, config=deployment_config)


if __name__ == "__main__":
    main()


# COMMAND ----------


