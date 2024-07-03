import os
import time
import sqlite3
import pandas as pd
from typing import List
from pymongo import MongoClient
from datetime import datetime

#
def load_data_tomongo(path, dbname, dbcollection):

    client = MongoClient("localhost", "27017")
    db = client[dbname]
    collection = db[dbcollection]

    # Load CSV file into DataFrame
    df = pd.read_csv(path)
    df["log_time"] = time.time()

    # Convert DataFrame to dictionary
    data = df.to_dict(orient="records")

    # Insert data into MongoDB collection
    collection.insert_many(data)

    print("Data imported successfully!")


def load_data_from_mongo(dbname, dbcollection):

    client = MongoClient("localhost", "27017")
    db = client[dbname]
    collection = db[dbcollection]

    data = collection.find()
    return data


def insert_collection(dbname, dbcollection, data):

    client = MongoClient("localhost", 27017)
    db = client[dbname]

    collection = db[dbcollection]
    update = {"$set": {"Metrics": data}}
    collection.update_one({}, update)


def create_df_table(dbname, tablename, dfpath):

    conn = sqlite3.connect(dbname)

    df = pd.read_csv(dfpath)
    df["log_time"] = time.time()

    df.to_sql(tablename, conn, if_exists="fail", index=False)
    conn.close()


def save_df(db_directory, dbname, dfname, dfpath=None, data=None):

    db_path = os.path.join(db_directory, dbname)
    conn = sqlite3.connect(db_path)

    now = datetime.now()
    formatted_date = now.strftime("%d/%B/%Y")

    if dfpath:

        df = pd.read_csv(dfpath)
        df["date"] = formatted_date
        df.to_sql(dfname, conn, if_exists="append", index=False)

    else:

        data["date"] = formatted_date
        data.to_sql(dfname, conn, if_exists="append", index=False)

    conn.close()


def insert_record(dbname, tablename, record: tuple):

    conn = sqlite3.connect(dbname)
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {tablename} {(record)}")

    conn.commit()
    conn.close()


def create_database_table(datapath, dbname, tablename):

    conn = sqlite3.connect(dbname)
    cursor = conn.cursor()

    df = pd.read_csv(datapath)
    df["log_time"] = time.time()

    column_dict = {}

    categorical_col = df.dtypes[df.dtypes == "object"].index.tolist()
    categorical_col = {col: "text" for col in categorical_col}
    column_dict.update(categorical_col)

    int_col = df.dtypes[df.dtypes == "int64"].index.tolist()
    int_col = {col: "int" for col in int_col}
    column_dict.update(int_col)

    float_col = df.dtypes[df.dtypes == "float64"].index.tolist()
    float_col = {col: "real" for col in float_col}
    column_dict.update(float_col)

    columns_str = ", ".join(
        [f"{col_name} {data_type}" for col_name, data_type in column_dict.items()]
    )

    cursor.execute(f"CREATE TABLE {tablename} ({columns_str})")

    conn.commit()
    conn.close()


def load_data(db_directory, dbname, tablename, filter_str=None):

    db_path = os.path.join(db_directory, dbname)
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {tablename}"

    # Load data into a DataFrame
    df = pd.read_sql(query, conn)
    return df


def read_dataset(filepath: str):

    dataframe = pd.read_csv(filepath)
    return dataframe


def prepare_dataset(dataframe: pd.DataFrame, target_col, drop_cols=None):

    dataframe.columns = dataframe.columns.str.replace(" ", "_").str.lower()
    categorical_col = dataframe.dtypes[dataframe.dtypes == "object"].index.tolist()

    for col in categorical_col:
        dataframe[col] = dataframe[col].str.replace(" ", "_").str.lower()

    dataframe = dataframe.drop(drop_cols, axis=1)
    dataframe = dataframe[dataframe["totalcharges"] != "_"]

    dataframe["totalcharges"] = dataframe["totalcharges"].astype("float32")
    dataframe[target_col] = (dataframe[target_col] == "yes").astype(int)

    return dataframe
