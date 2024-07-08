# Databricks notebook source
!pip install databricks-vectorsearch
!pip install -U mlflow
dbutils.library.restartPython()

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from mlflow.models import infer_signature
import sys
import os


EXAMPLE_QUESTION="Return the maximum and minimum number of cows across all farms."
EXAMPLE_MODEL_INPUT = {"prompt": [EXAMPLE_QUESTION]} 
VSC_INDEX = {"endpoint_name": "one-env-shared-endpoint-0", "index_name": "asong_demo.data.table_metadata_index"}
LLM_ENDPOINT="va_sqlcoder_7b_2"
REGISTERED_MODEL_NAME = "asong_dev.llms.text2sqlrag"

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
# os.environ['DATABRICKS_TOKEN']=mlflow.utils.databricks_utils.get_databricks_host_creds().token

class TextToSQLRAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self, vsc_index, llm_endpoint="sqlcoder_7b"):
        """
        Initialize the TextToSQLRAGModel.

        Parameters:
        vsc_index (dict): A dictionary containing the endpoint name and index name for Vector Search Client.
            - endpoint_name (str): The name of the Vector Search endpoint.
            - index_name (str): The name of the index in the Vector Search endpoint.
        llm_endpoint (str, optional): The name of the Language Model endpoint. Defaults to "sqlcoder_7b".
        """
        self.llm_endpoint = llm_endpoint
        self.client = get_deploy_client("databricks")
        self.vsc = VectorSearchClient(disable_notice=True,
                                      workspace_url=host, 
                                      personal_access_token=mlflow.utils.databricks_utils.get_databricks_host_creds().token)
        self.vsc_endpoint = vsc_index.get("endpoint_name")
        self.index_name = vsc_index.get("index_name")
        self.index = self.vsc.get_index(self.vsc_endpoint, self.index_name)

    def _build_prompt(self, question):
        """
        This method generates the prompt for the model.
        """
        TASK_KEY = "### Task"
        TASK = f"Generate a SQL query to answer [QUESTION]{question}[/QUESTION]"
        DATABASE_SCHEMA_KEY = "### Database Schema"
        DATABASE_SCHEMA = self._retrieve_database_context(
            question
        )  # todo retrieve from vector database
        ANSWER_KEY = "### Answer"
        ANSWER = f"Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]\n[SQL]"

        return f"""{TASK_KEY}
{TASK}
{DATABASE_SCHEMA_KEY}
{DATABASE_SCHEMA}
{ANSWER_KEY}
{ANSWER}
        """

    def _retrieve_database_context(self, question):
        """
        This method retrieves the database context for the given question.
        """
        results = self.index.similarity_search(
            query_text=question,
            columns=["TableName", "CreateTableStatement", "TableDescription"],
            num_results=5,
        )

        context = ""
        for res in results.get("result").get("data_array"):
            context += f"""TableName: {res[0]}
            CreateTableStatement: {res[1]}
            TableDescription: {res[2]}
            """
        return context

    def predict(self, context, model_input):
        """
        This method generates a prediction for the given input.
        """

        # NOTE: mlflow automatically converts dict inputs to a pandas dataframes so we must
        # convert the input back to a dict. this is expected to change in mlflow 3.0
        model_input = model_input.to_dict(orient="records")[0]
        question = model_input["prompt"][0]
        print(f"question:{question}")

        # Build the prompt

        ### 
        #Brian Comment: Add a breakpoint here and log out to make sure the question isn't being reformated? 
        ###
        prompt = self._build_prompt(question)

        # Generate response
        generated_response = self.client.predict(
            endpoint=self.llm_endpoint, inputs={"prompt": [prompt]}
        )
        generated_response = generated_response["choices"][0]["text"]
        generated_sql = generated_response[
            generated_response.find("[SQL]\n") + 7 : generated_response.find("</s>")
        ].strip()
        return {"generated_sql": [generated_sql]}

    
# def log_and_register_mlflow_model(model):
#     prediction = model.predict(context=None,model_input=pd.DataFrame(example_model_input))
#     signature = infer_signature(example_model_input, prediction)
    
#     mlflow.set_registry_uri("databricks-uc")
#     # mlflow.set_experiment("Workspace/Users/april@databricks.com/sqlcoder-7b")
    
#     with mlflow.start_run() as run:
#         mlflow.pyfunc.log_model(
#             artifact_path="model",
#             python_model=model,
#             input_example=example_model_input,
#             signature=signature,
#             registered_model_name=REGISTERED_MODEL_NAME, 
#             # example_no_conversion=True,
#         )

def main():

    model = TextToSQLRAGModel(vsc_index=VSC_INDEX, llm_endpoint=LLM_ENDPOINT)
    prediction = model.predict(context=None,model_input=pd.DataFrame(EXAMPLE_MODEL_INPUT))
    signature = infer_signature(EXAMPLE_MODEL_INPUT, prediction)
    
    mlflow.set_registry_uri("databricks-uc")
    # mlflow.set_experiment("Workspace/Users/april@databricks.com/sqlcoder-7b")
    
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            input_example=EXAMPLE_MODEL_INPUT,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME, 
            # example_no_conversion=True,
        )

if __name__ == "__main__":
    main()

    

# COMMAND ----------

# Name of the registered MLflow model
# registered_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
import sys
import os
sys.path.append(os.path.abspath('..'))

from text2sql_rag_model.model_deployment.utils import get_latest_model_version
lastest_version = get_latest_model_version(REGISTERED_MODEL_NAME)

EXAMPLE_QUESTION="Return the maximum and minimum number of cows across all farms."
example_model_input = {"prompt": [EXAMPLE_QUESTION]} 
loaded_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{lastest_version}")
loaded_model.predict(example_model_input)

# COMMAND ----------


