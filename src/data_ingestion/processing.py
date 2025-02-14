import os
import pandas as pd
from prefect import task, flow
from dotenv import load_dotenv
from utils import process_dataframe, pull_data_from_db, push_data_to_db

load_dotenv()
db_name = os.getenv("DB_NAME")
db_dir = os.getenv("DB_DIRECTORY")

dataset_path = os.getenv("PROCESSED_DATASET_PATH")
table_name = os.getenv("PROCESSED_DATASET_NAME")

drop_columns = os.getenv("DROP_COLUMNS")
target_column_name = os.getenv("TARGET_COLUMN")


# @flow(name="Data Processing")
def main():

    input_data = pull_data_from_db(tablename="RawData")
    processed_data = process_dataframe(
        input_data, target_column_name, drop_cols=drop_columns
    )
    push_data_to_db(
        tablename=table_name,
        data=processed_data,
    )


if __name__ == "__main__":
    main()
