from app.churn_guard.train_pipeline.train import training_pipeline
from app.churn_guard.utils.deploy import deploy_production


if __name__ == "__main__":
    training_pipeline()
    result = deploy_production()
    print("Succesful")
