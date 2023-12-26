import os
import pandas as pd
from utils_and_constants import RAW_DATASET, TARGET_COLUMN, PROCESSED_DATASET


def read_dataset(filepath: str):

    dataframe = pd.read_csv(filepath)
    dataframe.columns = dataframe.columns.str.replace(" ", "_").str.lower()

    categorical_col = dataframe.dtypes[dataframe.dtypes == "object"].index.tolist()
    for col in categorical_col:
        dataframe[col] = dataframe[col].str.replace(" ", "_").str.lower()

    return dataframe


def prepare_dataset(dataframe: pd.DataFrame):

    dataframe = dataframe[dataframe["totalcharges"] != "_"]
    dataframe["totalcharges"] = dataframe["totalcharges"].astype("float32")

    dataframe[TARGET_COLUMN] = (dataframe[TARGET_COLUMN] == "yes").astype(int)

    return dataframe


def main():
    # Read data
    churn_data = read_dataset(filepath=RAW_DATASET)
    churn_data = prepare_dataset(churn_data)

    if not os.path.exists(os.path.dirname(PROCESSED_DATASET)):
        os.mkdir(os.path.dirname(PROCESSED_DATASET))

    churn_data.to_csv(PROCESSED_DATASET, index=None)


if __name__ == "__main__":
    main()
