import os
from typing import List
import pandas as pd
from utils import (
    RAW_DATASET,
    TARGET_COLUMN,
    PROCESSED_DATASET,
    DROP_COLUMNS,
    DB_NAME,
    PROCESSED_DATASET_NAME,
    RAW_DATASET_NAME,
    DB_DIRECTORY,
)
from prefect import task, flow
from datahelper import save_df

# Load data
@task(name="Load Data")
def read_dataset(filepath: str):

    dataframe = pd.read_csv(filepath)
    save_df(DB_DIRECTORY, DB_NAME, RAW_DATASET_NAME, data=dataframe)
    return dataframe


# Prepare Data
@task(name="Process Data")
def prepare_dataset(dataframe: pd.DataFrame, drop_cols: List):

    dataframe.columns = dataframe.columns.str.replace(" ", "_").str.lower()

    categorical_col = dataframe.dtypes[dataframe.dtypes == "object"].index.tolist()
    for col in categorical_col:
        dataframe[col] = dataframe[col].str.replace(" ", "_").str.lower()

    dataframe = dataframe.drop(drop_cols, axis=1)
    dataframe = dataframe[dataframe["totalcharges"] != "_"]
    dataframe["totalcharges"] = dataframe["totalcharges"].astype("float32")
    dataframe[TARGET_COLUMN] = (dataframe[TARGET_COLUMN] == "yes").astype(int)

    return dataframe


@flow(name="Data Processing")
def main():
    churn_data = read_dataset(filepath=RAW_DATASET)
    churn_data = prepare_dataset(churn_data, drop_cols=DROP_COLUMNS)

    save_df(DB_DIRECTORY, DB_NAME, PROCESSED_DATASET_NAME, data=churn_data)

    # if not os.path.exists(os.path.dirname(PROCESSED_DATASET)):
    #     os.mkdir(os.path.dirname(PROCESSED_DATASET))

    # churn_data.to_csv(PROCESSED_DATASET, index=None)


if __name__ == "__main__":
    main()
