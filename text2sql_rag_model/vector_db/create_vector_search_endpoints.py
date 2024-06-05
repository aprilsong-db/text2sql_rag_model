# Databricks notebook source
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# vector search endpoint name
dbutils.widgets.text(
    "endpoint_name",
    f"",
    label="Vector Search Endpoint Name",
)


# COMMAND ----------


from databricks.vector_search.client import VectorSearchClient
from utils import endpoint_exists, wait_for_vs_endpoint_to_be_ready
import time

VECTOR_SEARCH_ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
if not VECTOR_SEARCH_ENDPOINT_NAME:
    raise Exception("Please specify a valid endpoint name")

vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

