# Databricks notebook source
# !pip install databricks-vectorsearch
# !pip install -U mlflow
# dbutils.library.restartPython()

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from mlflow.models import infer_signature
import sys
import os

EXAMPLE_QUESTION="Return the maximum and minimum number of cows across all farms."
# EXAMPLE_MODEL_INPUT = {"prompt": [EXAMPLE_QUESTION]} 
EXAMPLE_MODEL_INPUT = {"messages": [{"role": "user","content": EXAMPLE_QUESTION}]} 
VSC_INDEX = {"endpoint_name": "one-env-shared-endpoint-0", "index_name": "asong_demo.data.table_metadata_index"}
LLM_ENDPOINT="va_sqlcoder_7b_2"
REGISTERED_MODEL_NAME = "asong_dev.llms.text2sqlrag"

DATABRICKS_HOST = "https://e2-demo-field-eng.cloud.databricks.com"
DATABRICKS_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token
# os.environ['DATABRICKS_TOKEN']=mlflow.utils.databricks_utils.get_databricks_host_creds().token

class TextToSQLRAGModel(mlflow.pyfunc.PythonModel):
    @mlflow.trace(name="TextToSQLRAGModel__init__", span_type='func')
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
                                      workspace_url=DATABRICKS_HOST, 
                                      personal_access_token=DATABRICKS_TOKEN)
        self.vsc_endpoint = vsc_index.get("endpoint_name")
        self.index_name = vsc_index.get("index_name")
        self.index = self.vsc.get_index(self.vsc_endpoint, self.index_name)
    
    @mlflow.trace(name="_retrieve_database_context", span_type='RETREIVER')
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
    
    @mlflow.trace(name="_build_prompt", span_type='func')
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
    @mlflow.trace(name="_parse_response", span_type='func')
    def _parse_response(self, response):
        generated_sql = response["choices"][0]["text"].strip()
        return generated_sql

    @mlflow.trace(name="_generate_response", span_type='func')
    def _generate_response(self, prompt):
        generated_response = self.client.predict(
            endpoint=self.llm_endpoint, inputs={"prompt": [prompt]}
        )
        generated_sql = self._parse_response(generated_response)
        return {"generated_sql": [generated_sql]}

    @mlflow.trace(name="text2sql-agent-predict", span_type='func')
    def predict(self, context, model_input):
        """
        This method generates a prediction for the given input.
        """

        # NOTE: mlflow automatically converts dict inputs to a pandas dataframes so we must
        # convert the input back to a dict. this is expected to change in mlflow 3.0
        # to allow working with dict inputs in development before logging, we will convert input to pandas dataframe mimicking noted mlflow behavior
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        question = model_input.to_dict(orient='records')[0].get('messages')[0].get('content')

        # Build the prompt
        prompt = self._build_prompt(question)

        # Generate response
        generated_response = self._generate_response(prompt)
        return generated_response

# COMMAND ----------

chain = TextToSQLRAGModel(vsc_index=VSC_INDEX, llm_endpoint=LLM_ENDPOINT)
prediction = chain.predict(context=None,model_input=EXAMPLE_MODEL_INPUT)

# COMMAND ----------

mlflow.models.set_model(model=chain)

# COMMAND ----------


