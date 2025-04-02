import os
import yaml
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from utils import push_data_to_db, process_dataframe, pull_data_from_db, connect_sqlite


load_dotenv()

config_path = os.getenv("config_path")
with open(config_path) as config:
    config = yaml.safe_load(config)

customer_data_path = config["database"]["customer"]["database_path"]
db_engine = connect_sqlite(customer_data_path)

app = Flask("Data Ingestion and Processing")


def ingest():

    data = config["data"]
    data_path = data["raw"]["path"]
    table_name = data["raw"]["name"]

    logging.info("Ingesting data to database")
    push_data_to_db(db_engine, tablename=table_name, dfpath=data_path)
    logging.info("Ingestion Successful")


def process():

    data = config["data"]
    raw_table_name = data["raw"]["name"]
    processed_table_name = data["process"]["name"]
    drop_columns = data["process"]["dropcols"]
    target_column = data["process"]["targetcolumn"]

    input_data = pull_data_from_db(db_engine, tablename=raw_table_name)
    processed_data = process_dataframe(
        input_data, target_column, drop_cols=drop_columns
    )
    push_data_to_db(
        db_engine,
        tablename=processed_table_name,
        data=processed_data,
    )


@app.route("/process", methods=["GET"])
def main():

    ingest()
    process()
    return jsonify({"response": "Successfully Processes Data"})


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
