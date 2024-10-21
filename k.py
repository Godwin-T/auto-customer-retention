import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


dbname = os.getenv("DBNAME")
# username = os.getenv("MYSQL_USERNAME")
# password = os.getenv("MYSQL_PASSWORD")
# hostname = os.getenv("THOSTNAME")

if dbname:
    print(f"API Key: {dbname}")
else:
    print("API Key not found!")

# engine = create_engine(
#     f"mysql+mysqlconnector://{username}:{password}@{hostname}/{dbname}"
# )


# def load_data_from_mysql_db(sql_engine, tablename):

#     query = f"SELECT * FROM {tablename}"
#     df = pd.read_sql(query, con=sql_engine)
#     return df


# df = load_data_from_mysql_db(engine, "ProcessedData")
# print("True")
