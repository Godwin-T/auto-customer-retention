import os
import yaml
import logging
import argparse

# from prefect import task, flow
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from utils import push_data_to_db, process_dataframe, pull_data_from_db


load_dotenv()
config_path = os.getenv("config_path")

app = Flask("Data Ingestion and Processing")

logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config(config_path):

    global config
    with open(config_path) as config:
        config = yaml.safe_load(config)


def ingest():

    data = config["data"]
    data_path = data["raw"]["path"]
    table_name = data["raw"]["name"]

    logging.info("Ingesting data to database")
    push_data_to_db(tablename=table_name, dfpath=data_path)
    logging.info("Ingestion Successful")


def process():

    data = config["data"]
    raw_table_name = data["raw"]["name"]
    processed_table_name = data["process"]["name"]
    drop_columns = data["process"]["dropcols"]
    target_column = data["process"]["targetcolumn"]

    logging.info("Pulling from database")
    input_data = pull_data_from_db(tablename=raw_table_name)
    logging.info("Process data")
    processed_data = process_dataframe(
        input_data, target_column, drop_cols=drop_columns
    )
    logging.info("Push to database")
    push_data_to_db(
        tablename=processed_table_name,
        data=processed_data,
    )
    logging.info("Successfully Processed Data")


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None, help="Database hostname")
    parser.add_argument("--dbname", default=None, help="Database name")
    parser.add_argument("--username", default=None, help="Database username")
    parser.add_argument("--passkey", default=None, help="Database password")
    parser.add_argument("--config", dest="config", required=True)

    args = parser.parse_args()
    return args


@app.route("/process", methods=["GET"])
def main():
    load_config(config_path)
    ingest()
    process()
    return jsonify({"response": "Successfully Processes Data"})


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
