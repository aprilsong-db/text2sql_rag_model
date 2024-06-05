
# Databricks notebook source

%pip install databricks-vectorsearch
dbutils.library.restartPython()

import sys
from databricks.vector_search.client import VectorSearchClient

def main():
    vsc = VectorSearchClient()
    endpoint_name = sys.argv[1]  # Get the endpoint name from the command line arguments

    try:
        vsc.create_endpoint(name=endpoint_name,
                            endpoint_type="STANDARD")
    except Exception as e:
        if "already exists" in str(e):
            pass
        else:
            raise e

if __name__ == "__main__":
    main()