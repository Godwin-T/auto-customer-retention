#!/bin/bash

# Define the path to the host directory and the container directory
HOST_RAW_DATASET="$(pwd)/../data/churn-data/raw_data/Telco-Customer-Churn.csv"
CONTAINER_RAW_DATASET="/home/data/raw_data/Churn.csv"

HOST_MODEL_DATABASE="$(pwd)/../../databases/mlflow.db"
CONTAINER_MODEL_DATABASE="/home/databases/mlflow.db"

HOST_DATA_DATABASE="$(pwd)/../../databases/"
CONTAINER_DATA_DATABASE="/home/databases/"

HOST_MODEL_PATH="$(pwd)/../models/churn_model.pkl"
CONTAINER_MODEL_PATH="/home/models/churn_model.pkl"


# Build the Docker image
docker build -t myapp:v1 .

# Run the Docker container with the host volume
docker run -v $HOST_RAW_DATASET:$CONTAINER_RAW_DATASET -v $HOST_DATA_DATABASE:$CONTAINER_DATA_DATABASE -v $HOST_MODEL_DATABASE:$CONTAINER_MODEL_DATABASE -v $HOST_MODEL_PATH:$CONTAINER_MODEL_PATH myapp:v1
