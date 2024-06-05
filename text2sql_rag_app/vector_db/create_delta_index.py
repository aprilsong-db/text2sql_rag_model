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

# source table name
dbutils.widgets.text(
    "source_table_name",
    f"",
    label="Source Table Name",
)

# index name
dbutils.widgets.text(
    "index_name",
    f"",
    label="Index Name",
)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from utils import index_exists, wait_for_index_to_be_ready

vsc = VectorSearchClient()

endpoint_name = dbutils.widgets.get("endpoint_name")
source_table_name = dbutils.widgets.get("source_table_name")
index_name = dbutils.widgets.get("index_name")

if not index_exists(vsc, endpoint_name, index_name):
    print(f"Creating index {index_name} on endpoint {endpoint_name}...")
    index = vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name=source_table_name,
        index_name=index_name,
        pipeline_type='TRIGGERED',
        primary_key="TableName",
        embedding_source_column="CreateTableStatement",
        embedding_model_endpoint_name="databricks-bge-large-en"
    )

else:
    #Trigger a sync to update our vs content with the new data saved in the table
    print(f"Index {index_name} already exists. Triggering a sync to update our vs content")
    vsc.get_index(endpoint_name, index_name).sync()

# COMMAND ----------

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, endpoint_name, index_name)
print(f"index {index_name} on table {source_table_name} is ready")

# COMMAND ----------


