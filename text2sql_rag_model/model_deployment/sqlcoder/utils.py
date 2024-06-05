from mlflow import MlflowClient

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