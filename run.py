from app.churn_guard.train_pipeline.train import training_pipeline
from app.churn_guard.utils.deploy import deploy_production
from app.churn_guard.utils.process_data import main

training_pipeline()
# print("Successful")
result = deploy_production()
# print("Successful")
