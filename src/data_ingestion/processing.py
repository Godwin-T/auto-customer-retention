import os
import pandas as pd
from prefect import task, flow
from dotenv import load_dotenv
from utils import save_dataframe_to_relational_db, load_data_from_relational_db

load_dotenv()

db_name = os.getenv("DB_NAME")
db_dir = os.getenv("DB_DIRECTORY")

dataset_path = os.getenv("PROCESSED_DATASET_PATH")
dataset_name = os.getenv("PROCESSED_DATASET_NAME")

drop_columns = os.getenv("DROP_COLUMNS")
target_column_name = os.getenv("TARGET_COLUMN")


# Prepare Data
# @task(name="Process raw data")
def process_dataset(dataframe: pd.DataFrame, target_column_name, drop_cols=None):

    dataframe.columns = dataframe.columns.str.replace(" ", "_").str.lower()

    categorical_col = dataframe.dtypes[dataframe.dtypes == "object"].index.tolist()
    for col in categorical_col:
        dataframe[col] = dataframe[col].str.replace(" ", "_").str.lower()

    # dataframe = dataframe.drop([drop_cols], axis=1)
    dataframe = dataframe[dataframe["totalcharges"] != "_"]
    dataframe.loc[:, "totalcharges"] = dataframe["totalcharges"].astype(
        "float32"
    )  # "totalcharges"
    dataframe.loc[:, target_column_name] = (
        dataframe[target_column_name] == "yes"
    ).astype(
        int
    )  # target_column_name

    return dataframe


# @flow(name="Data Processing")
def main():

    input_data = load_data_from_relational_db(
        dbprovider="sqlite", db_directory=db_dir, dbname=db_name, tablename="RawData"
    )
    churn_data = process_dataset(input_data, target_column_name, drop_cols=drop_columns)
    save_dataframe_to_relational_db(
        tablename=dataset_name,
        dbprovider="sqlite",
        data=churn_data,
        db_directory=db_dir,
        dbname=dataset_name,
    )


if __name__ == "__main__":
    main()
