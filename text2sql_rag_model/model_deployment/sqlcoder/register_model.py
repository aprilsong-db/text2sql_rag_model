# Databricks notebook source
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


HF_MODEL_NAME = "defog/sqlcoder-7b-2" # Specify the name of the pre-trained model to be used
TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_NAME) # Load the tokenizer associated with the pre-trained model
EOS_TOKEN_ID = TOKENIZER.eos_token_id

MODEL_CONFIG = {
        "num_beams": 1,
        "max_new_tokens": 400,
        "do_sample": False,
        "return_full_text": False,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": EOS_TOKEN_ID,
        "num_return_sequences":1,
    }
REGISTERED_MODEL_NAME = "asong_dev.llms.sqlcoder_7b"
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

def load_model():
    # Check if CUDA is available for GPU acceleration
    torch.cuda.is_available()

    # Get the total available memory on the GPU device
    available_memory = torch.cuda.get_device_properties(0).total_memory

    # Choose the appropriate model configuration based on available GPU memory
    if available_memory > 15e9:
        # If GPU memory is greater than 15GB, load the model in float16 (bfloat16) for faster processing
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )
    else:
        # If GPU memory is less than or equal to 15GB, load the model in 8 bits for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            trust_remote_code=True,
            load_in_8bit=True,
            use_cache=True,
        )
    
    return model

def log_and_register_mlflow_model(model):
    prediction = model(EXAMPLE_PROMPT)
    signature = infer_signature(EXAMPLE_PROMPT, prediction)
    
    mlflow.set_registry_uri("databricks-uc")
    # mlflow.set_experiment("Workspace/Users/april@databricks.com/sqlcoder-7b")
    
    with mlflow.start_run() as run:

        mlflow.transformers.log_model(
            task="text-generation",
            transformers_model=model,
            model_config=MODEL_CONFIG,
            artifact_path="model",
            signature=signature,
            input_example=EXAMPLE_PROMPT,
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={"task": "llm/v1/completions"},
            example_no_conversion=True, #the input example will not be converted to a Pandas DataFrame format when saving the model
        )

def main():

    model = load_model()
    
    text2sql_pipe = pipeline(task="text-generation",
        model=model,
        tokenizer=TOKENIZER,
        **MODEL_CONFIG,
    )
    
    log_and_register_mlflow_model(text2sql_pipe)


if __name__ == "__main__":
    main()
