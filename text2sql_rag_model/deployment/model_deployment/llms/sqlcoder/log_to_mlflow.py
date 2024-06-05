import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import mlflow
from mlflow.models import infer_signature


MODEL_NAME = "defog/sqlcoder-7b-2" # Specify the name of the pre-trained model to be used
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME) # Load the tokenizer associated with the pre-trained model
EOS_TOKEN_ID = TOKENIZER.eos_token_id

def load_model():
    # Check if CUDA is available for GPU acceleration
    torch.cuda.is_available()

    # Get the total available memory on the GPU device
    available_memory = torch.cuda.get_device_properties(0).total_memory

    # Choose the appropriate model configuration based on available GPU memory
    if available_memory > 15e9:
        # If GPU memory is greater than 15GB, load the model in float16 (bfloat16) for faster processing
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )
    else:
        # If GPU memory is less than or equal to 15GB, load the model in 8 bits for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            load_in_8bit=True,
            use_cache=True,
        )
    
    return model

def main():

    model = load_model()
    
    model_config = {
        "num_beams": 1,
        "max_new_tokens": 400,
        "do_sample": False,
        "return_full_text": False,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": EOS_TOKEN_ID,
        "num_return_sequences":1,
    }

    text2sql_pipe = pipeline(task="text-generation",
        model=model,
        tokenizer=TOKENIZER,
        **model_config,
    )

    question="Return the maximum and minimum number of cows across all farms."
    input_prompt = f"""
    ### Task
    Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

    ### Database Schema
    The query will run on a database with the following schema:
    CREATE TABLE farm (\n  Farm_ID BIGINT,\n  Year BIGINT,\n  Total_Horses DOUBLE,\n  Working_Horses DOUBLE,\n  Total_Cattle DOUBLE,\n  Oxen DOUBLE,\n  Bulls DOUBLE,\n  Cows DOUBLE,\n  Pigs DOUBLE,\n  Sheep_and_Goats DOUBLE)\nUSING delta\nCOMMENT 'The \\'farm\\' table contains data related to various farm animals. It includes information such as the total number of horses, cattle, oxen, and other livestock for a particular year. This data can be useful for analyzing trends in animal populations over dtime and planning future farming strategies based on current stock levels.

    CREATE TABLE farm_competition (\n  Competition_ID BIGINT,\n  Year BIGINT,\n  Theme STRING,\n  Host_city_ID BIGINT,\n  Hosts STRING)\nUSING delta\nCOMMENT 'The \\'farm_competition\\' table contains information about various agricultural competitions held across different cities. It includes details such as the theme of the competition, the year it was held, and the host city and its respective hosts. This data can be useful for understanding trends in agricultural competitions over the years, analyzing the popularity of different themes among competitors, and studying the geographical distribution of these events.

    ### Answer
    Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
    [SQL]
    """
    prediction = text2sql_pipe(input_prompt)
    
    signature = infer_signature(input_prompt, prediction)
    
    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run() as run:

        mlflow.transformers.log_model(
            task="text-generation",
            transformers_model=text2sql_pipe,
            model_config=model_config,
            artifact_path="model",
            signature=signature,
            input_example=input_prompt,
            registered_model_name="asong_dev.llms.sqlcoder_7b",
            metadata={"task": "llm/v1/completions"},
            example_no_conversion=True, #the input example will not be converted to a Pandas DataFrame format when saving the model
        )

if __name__ == "__main__":
    main()