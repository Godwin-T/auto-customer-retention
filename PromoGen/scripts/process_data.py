import os
import pandas as pd
from prefect import task, flow
from constant import RAW_DATAPATH, DATA_PATH


@task(name="Load Data")
def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(" ", "_").str.lower()
    return data


@flow(name="Data Processing")
def main():

    dataframe = pd.DataFrame(columns=["context", "title", "body"])

    for mail in os.listdir(RAW_DATAPATH):
        df_path = os.path.join(RAW_DATAPATH, mail)
        df = load_data(df_path)
        dataframe = pd.concat([dataframe, df])

    if not os.path.exists(os.path.dirname(DATA_PATH)):
        os.mkdir(os.path.dirname(DATA_PATH))

    dataframe.to_csv(DATA_PATH, index=False)
    print("Processed Data Successful")


if __name__ == "__main__":
    main()
