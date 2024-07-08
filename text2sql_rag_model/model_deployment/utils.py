from mlflow import MlflowClient
import requests
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from mlflow.deployments import get_deploy_client
from mlflow.utils.databricks_utils import get_databricks_host_creds
from databricks.sdk.runtime import *


def load_model_and_tokenizer(pretrained_model_name_or_path):
    """
    Load the pretrained model and tokenizer from Hugging Face.
    """
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path) 
        
    return model, tokenizer


def get_latest_model_version(model_name):
    """Get latest model version, used for POC demo.

    Args:
        model_name (str): Name of ML model

    Returns:
        int: latest version of model for given name
    """
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version




def get_api_root_and_token():
    """
    Get the API root URL and API token from the Databricks notebook context.
    
    Returns:
    - API_ROOT (str): The API root URL.
    - API_TOKEN (str): The API token.
    """
    API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
    API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    return API_ROOT, API_TOKEN

def get_optimizable_info(registered_model_name, latest_model_version):
    """
    Get the optimization information for a registered model and its latest version.
    
    Args:
    - registered_model_name (str): The name of the registered model.
    - latest_model_version (int): The latest version of the model.
    
    Returns:
    - optimizable_info (dict): The optimization information for the model.
    """
    API_ROOT, API_TOKEN = get_api_root_and_token()
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}
    optimizable_info = (requests.get(
        url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{registered_model_name}/{latest_model_version}",
        headers=headers)
        .json())
    return optimizable_info

def get_max_provisioned_throughput(registered_model_name, latest_model_version):
    """
    Get the maximum provisioned throughput for a registered model and its latest version.
    
    Args:
    - registered_model_name (str): The name of the registered model.
    - latest_model_version (int): The latest version of the model.
    
    Returns:
    - max_provisioned_throughput (int): The maximum provisioned throughput for the model.
    
    Raises:
    - ValueError: If the model is not eligible for provisioned throughput.
    """
    optimizable_info = get_optimizable_info(registered_model_name, latest_model_version)
    if 'optimizable' not in optimizable_info or not optimizable_info['optimizable']:
        raise ValueError("Model is not eligible for provisioned throughput")
    chunk_size = optimizable_info['throughput_chunk_size']
    max_provisioned_throughput = 2 * chunk_size
    return max_provisioned_throughput



def create_or_update_model_endpoint(name, config):
    deploy_client = get_deploy_client("databricks")

    try:
        print(f"""Attempting to create endpoint {name} with model version {config.get('served_entities')[0].get("entity_version")}""")
        endpoint = deploy_client.create_endpoint(
            name=name,
            config=config
        )
    except Exception as e:
        print(f"Endpoint creation failed with error: {e}, \nattempting to update endpoint...")
        try:
            endpoint = deploy_client.update_endpoint(
                endpoint=name,
                config=config
            )
            #todo: wait for endpoint to be ready
        except Exception as e:
            print(f"Endpoint update failed with error: {e}")
            raise e
        #todo: wait for endpoint to be ready






# gather other inputs the API needs - they are used as environment variables in the
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

def endpoint_exists(serving_endpoint_name):
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint(serving_endpoint_name):
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint(serving_endpoint_name, served_models):
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {serving_endpoint_name}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": serving_endpoint_name, "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")
  
def update_endpoint(serving_endpoint_name, served_models):
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {serving_endpoint_name}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")