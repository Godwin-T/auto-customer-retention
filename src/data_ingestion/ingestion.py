import os
import pandas as pd
from prefect import task, flow
from dotenv import load_dotenv
from utils import push_data_to_db

load_dotenv()

db_name = os.getenv("DB_NAME")
db_dir = os.getenv("DB_DIRECTORY")

dfpath = os.getenv("RAW_DATASET_PATH")
table_name = os.getenv("RAW_DATASET_NAME")


# @flow(name="Pull Data from source")
def main():

    print("stargint")
    push_data_to_db(tablename=table_name, dfpath=dfpath)


# if __name__ == "__main_":
#     print("=================================")
#     main()
#     print("Successful")
main()
