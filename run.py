from src.backend.churn_guard import training_pipeline
from src.backend.churn_guard import deploy_production

if __name__ == "__main__":
    training_pipeline()
    result = deploy_production()
    print("Succesful")
