import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


dbname = os.getenv("DBNAME")
username = os.getenv("MYSQL_USERNAME")
password = os.getenv("MYSQL_PASSWORD")
hostname = os.getenv("HOSTNAME")
dataname = os.getenv("PROCESSED_DATASET_NAME")

if dbname:
    print(f"API Key: {dbname}")
else:
    print(f"API Key {dbname} not found!")

if username:
    print(f"API Key: {username}")
else:
    print(f"API Key {username} not found!")

if password:
    print(f"API Key: {password}")
else:
    print(f"API Key {password} not found!")

if hostname:
    print(f"API Key: {hostname}")
else:
    print(f"API Key {hostname} not found!")

if hostname:
    print(f"API Key: {hostname}")
else:
    print(f"API Key {hostname} not found!")


if dataname:
    print(f"API Key: {dataname}")
else:
    print(f"API Key {dataname} not foud!")


# engine = create_engine(
#     f"mysql+mysqlconnector://{username}:{password}@{hostname}/{dbname}"
# )


# def load_data_from_mysql_db(sql_engine, tablename):

#     query = f"SELECT * FROM {tablename}"
#     df = pd.read_sql(query, con=sql_engine)
#     return df


# df = load_data_from_mysql_db(engine, "ProcessedData")
# print("True")
