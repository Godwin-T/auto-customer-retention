# ingestion.py
import os
import yaml
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from utils import push_data_to_db, process_dataframe, pull_data_from_db, connect_sqlite

load_dotenv()


def load_config():
    config_path = os.getenv("config_path")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_db_engine():
    config = load_config()
    return connect_sqlite(config["database"]["customer"]["database_path"])


def create_app(engine=None):
    app = Flask("Data Ingestion and Processing")

    if engine is None:
        engine = get_db_engine()

    @app.route("/process", methods=["GET"])
    def main():
        ingest(engine)
        process(engine)
        return jsonify({"response": "Successfully Processed Data"})

    return app


def ingest(engine):
    config = load_config()
    data = config["data"]
    data_path = data["raw"]["path"]
    table_name = data["raw"]["name"]

    logging.info("Ingesting data to database")
    push_data_to_db(engine, tablename=table_name, dfpath=data_path)
    logging.info("Ingestion Successful")


def process(engine):
    config = load_config()
    data = config["data"]
    raw_table_name = data["raw"]["name"]
    processed_table_name = data["process"]["name"]
    drop_columns = data["process"]["dropcols"]
    target_column = data["process"]["targetcolumn"]

    input_data = pull_data_from_db(engine, tablename=raw_table_name)
    processed_data = process_dataframe(
        input_data, target_column, drop_cols=drop_columns
    )
    push_data_to_db(engine, tablename=processed_table_name, data=processed_data)


# Only run the app directly if main
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8000, host="0.0.0.0")
