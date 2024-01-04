import os
import pandas as pd
from constant import RAW_DATAPATH, DATA_PATH


def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(" ", "_").str.lower()
    return data


def compile(data_path):
    dataframe = pd.DataFrame(columns=["context", "title", "body"])
    for mail in os.listdir(data_path):
        df_path = os.path.join(data_path, mail)
        df = load_data(df_path)
        dataframe = pd.concat([dataframe, df])
    return dataframe


def main():

    dataframe = compile(RAW_DATAPATH)

    if not os.path.exists(os.path.dirname(DATA_PATH)):
        os.mkdir(os.path.dirname(DATA_PATH))

    dataframe.to_csv(DATA_PATH, index=False)
    print("Processed Data Successful")


if __name__ == "__main__":
    main()
