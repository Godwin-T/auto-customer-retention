#!/bin/sh

# Run the data processing script
python process_data.py

# Run the training script
python train.py

#mlflow server --backend-store-uri sqlite:///./databases/mlflow.db --default-artifact-root file:///./databases/mlruns/artifacts/model --host 0.0.0.0 --port 5000
