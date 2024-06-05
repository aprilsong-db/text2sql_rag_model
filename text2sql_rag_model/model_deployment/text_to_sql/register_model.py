# Databricks notebook source
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch mlflow[gateway]==2.13.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow.pyfunc
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from mlflow.models import infer_signature


EXAMPLE_QUESTION="Return the maximum and minimum number of cows across all farms."
EXAMPLE_MODEL_INPUT = {"prompt": [EXAMPLE_QUESTION]} 
VSC_INDEX = {"endpoint_name": "one-env-shared-endpoint-0", "index_name": "asong_demo.data.table_metadata_index"}
LLM_ENDPOINT="va_sqlcoder_7b_2"
REGISTERED_MODEL_NAME = "asong_dev.llms.sqlcoder_7b"

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
        self.vsc = VectorSearchClient()
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

    def predict(self, context, model_input, params=None):
        """
        This method generates a prediction for the given input.
        """
        prompt = model_input["prompt"][0]

        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Generate response
        generated_response = self.client.predict(
            endpoint=self.llm_endpoint, inputs={"prompt": [prompt]}
        )
        generated_response = generated_response["choices"][0]["text"]
        generated_sql = generated_response[
            generated_response.find("[SQL]\n") + 7 : generated_response.find("</s>")
        ].strip()
        return {"generated_sql": [generated_sql]}
    
def log_and_register_mlflow_model(model):
    prediction = model.predict(context=None,model_input=EXAMPLE_MODEL_INPUT)
    signature = infer_signature(EXAMPLE_MODEL_INPUT, prediction)
    
    mlflow.set_registry_uri("databricks-uc")
    # mlflow.set_experiment("Workspace/Users/april@databricks.com/sqlcoder-7b")
    
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            input_example=EXAMPLE_MODEL_INPUT,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME
        )

def main():

    model = TextToSQLRAGModel(vsc_index=VSC_INDEX, llm_endpoint=LLM_ENDPOINT)
    log_and_register_mlflow_model(model)


if __name__ == "__main__":
    main()

    
