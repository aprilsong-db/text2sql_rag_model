# Databricks notebook source
!pip install -U -qqqq databricks-agents mlflow databricks-vectorsearch
dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
from databricks import agents

# Use the Unity Catalog model registry
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# Create widgets 
dbutils.widgets.text("uc_catalog", "asong_dev", "catalog")
dbutils.widgets.text("uc_schema", "llms", "schema")
dbutils.widgets.text("model_name", "text2sqlrag", "Model name")

# Retrieve the values from the widgets
uc_catalog = dbutils.widgets.get("uc_catalog")
uc_schema = dbutils.widgets.get("uc_schema")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

EXAMPLE_MODEL_INPUT = {
    "messages": [
        {
            "role": "user",
            "content": "Return the maximum and minimum number of cows across all farms.",
        }
    ]
}


# Specify the full path to the chain notebook
chain_notebook_file = "chain"
chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)

print(f"Chain notebook path: {chain_notebook_path}")

# COMMAND ----------

# MAGIC %md ## Log the chain
# MAGIC

# COMMAND ----------

with mlflow.start_run():
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,
        artifact_path="chain",
        input_example=EXAMPLE_MODEL_INPUT,
    )

# COMMAND ----------

# Unity Catalog location
uc_model_name = f"{uc_catalog}.{uc_schema}.{model_name}"

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=uc_model_name )
