import os
import pandas as pd
from prefect import task, flow
from dotenv import load_dotenv
from utils import save_dataframe_to_relational_db

load_dotenv()

db_name = os.getenv("DB_NAME")
db_dir = os.getenv("DB_DIRECTORY")

dataset_path = os.getenv("RAW_DATASET_PATH")
dataset_name = os.getenv("RAW_DATASET_NAME")


# Load data
# @task(name="Load data from path or bucket")
def load_dataset(filepath: str):

    dataframe = pd.read_csv(filepath)
    return dataframe


# @flow(name="Pull Data from source")
def main():

    input_data = load_dataset(filepath=dataset_path)
    save_dataframe_to_relational_db(
        tablename=dataset_name,
        dbprovider="sqlite",
        data=input_data,
        db_directory=db_dir,
        dbname=dataset_name,
    )


if __name__ == "__main_":
    main()
