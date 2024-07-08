# Databricks notebook source
# Update/Install required dependencies
!pip install -U mlflow
!pip install -U transformers
!pip install -U accelerate
!pip install -U pytest
dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
import torch
import mlflow
import numpy as np

sys.path.append(os.path.abspath('..'))

from text2sql_rag_model.model_deployment.utils import load_model_and_tokenizer
from config import CATALOG, SCHEMA, MODEL_NAME, REGISTERED_MODEL_NAME



HF_MODEL_NAME = "defog/sqlcoder-7b-2" 
QUESTION="Return the maximum and minimum number of cows across all farms."
EXAMPLE_PROMPT = f"""
### Task
Generate a SQL query to answer [QUESTION]{QUESTION}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
CREATE TABLE farm (\n  Farm_ID BIGINT,\n  Year BIGINT,\n  Total_Horses DOUBLE,\n  Working_Horses DOUBLE,\n  Total_Cattle DOUBLE,\n  Oxen DOUBLE,\n  Bulls DOUBLE,\n  Cows DOUBLE,\n  Pigs DOUBLE,\n  Sheep_and_Goats DOUBLE)\nUSING delta\nCOMMENT 'The \\'farm\\' table contains data related to various farm animals. It includes information such as the total number of horses, cattle, oxen, and other livestock for a particular year. This data can be useful for analyzing trends in animal populations over dtime and planning future farming strategies based on current stock levels.

CREATE TABLE farm_competition (\n  Competition_ID BIGINT,\n  Year BIGINT,\n  Theme STRING,\n  Host_city_ID BIGINT,\n  Hosts STRING)\nUSING delta\nCOMMENT 'The \\'farm_competition\\' table contains information about various agricultural competitions held across different cities. It includes details such as the theme of the competition, the year it was held, and the host city and its respective hosts. This data can be useful for understanding trends in agricultural competitions over the years, analyzing the popularity of different themes among competitors, and studying the geographical distribution of these events.

### Answer
Given the database schema, here is the SQL query that [QUESTION]{QUESTION}[/QUESTION]
[SQL]
"""
EXAMPLE_MODEL_INPUT = {"prompt": np.array([EXAMPLE_PROMPT])} 



def main():
    """
    Main function to load the model, set the model configuration, and log/register the MLflow model.
    """
    model, tokenizer = load_model_and_tokenizer(pretrained_model_name_or_path=HF_MODEL_NAME)
    
    # Set the model configuration
    eos_token_id = tokenizer.eos_token_id
    model_config = {
            "num_beams": 1,
            "max_new_tokens": 400,
            "do_sample": False,
            "eos_token_id": eos_token_id,
            "pad_token_id": eos_token_id,
            "num_return_sequences":1,
        }
    
    mlflow.set_registry_uri("databricks-uc")
    
    with mlflow.start_run() as run:
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        run = mlflow.transformers.log_model(
            task="llm/v1/completions",
            transformers_model=components,
            model_config=model_config,
            artifact_path="model",
            input_example=EXAMPLE_MODEL_INPUT,
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={"task": "llm/v1/completions"},
            example_no_conversion=True, #the input example will not be converted to a Pandas DataFrame format when saving the model
        )


if __name__ == "__main__":
    main()

# COMMAND ----------

import mlflow.pyfunc
from text2sql_rag_model.model_deployment.utils import get_latest_model_version

# Name of the registered MLflow model
registered_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"


lastest_version = get_latest_model_version(registered_model_name)
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/{lastest_version}")
# Make a prediction using the loaded model
prediction = loaded_model.predict(
    {
        "prompt": EXAMPLE_PROMPT,
    }
)
print(prediction)


# Example response
# [{'id': 'e1219e4d-0230-4aad-a91a-2d688b6054ee',
#   'object': 'text_completion',
#   'created': 1718663695,
#   'model': 'defog/sqlcoder-7b-2',
#   'usage': {'prompt_tokens': 429,
#    'completion_tokens': 32,
#    'total_tokens': 461},
#   'choices': [{'index': 0,
#     'finish_reason': 'stop',
#     'text': 'SELECT MAX(f.Cows) AS max_cows, MIN(f.Cows) AS min_cows FROM farm f;'}]}]

# COMMAND ----------


