from prefect import flow
from app.backend.churn_guard import data_processing_pipeline
from app.backend.churn_guard import training_pipeline


@flow
def main():

    data_processing_pipeline()
    training_pipeline()


if __name__ == "__main__":

    main.serve(name="data=processing-deployment", cron="0 */72 * * *")

    # main.deploy(
    #     name="data=processing-deployment",
    #     cron="*/1 * * * *",
    #     work_pool_name="customer-retention",
    # )
