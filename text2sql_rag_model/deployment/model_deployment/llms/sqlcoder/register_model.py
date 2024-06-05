from mlflow.tracking import MlflowClient
import mlflow

MODEL_NAME = "asong_dev.llms.sqlcoder_7b"

def main():
    # Instantiate the client
    client = MlflowClient()

    # Get a list of all model versions for a given registered model name
    model_version_infos = client.search_model_versions(f"name = '{MODEL_NAME}'")

    # Retrieve the latest model version number
    latest_model_version = max([model_version_info.version for model_version_info in model_version_infos])

    latest_model_version
    mlflow.set_registry_uri("databricks-uc")

if __name__ == "__main__":
    main()

    