"""
Routes for data ingestion component
"""

import logging
import pandas as pd
import uuid
from flask import request, jsonify

from common.database import get_db_engine
from config import load_config
from utils import pull_data_from_db, push_data_to_db
from .utils import process_dataframe, process_streamlit_dataframe


def process_existing_data():
    """Process data already in the database"""

    engine = get_db_engine()
    ingest(engine)
    process(engine)
    return jsonify(
        {"status": "success", "message": "Successfully processed existing data"}
    )


def upload_data():
    """Endpoint to receive data uploads from Streamlit"""

    engine = get_db_engine()
    # Check if a file was uploaded
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files["file"]

    # Check if file is empty
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    # Read the file into a pandas DataFrame
    if file and file.filename.endswith(".csv"):
        # Read CSV file into DataFrame
        df = pd.read_csv(file)

        # 1. Generate a new unique job_id
        job_id = str(uuid.uuid4())

        # 2. Add job_id column to the DataFrame
        df["job_id"] = job_id

        # Get table name from config
        config = load_config()
        table_name = config["data"]["streamlit"]["unprocessed_data"]

        # Push data to database
        push_data_to_db(engine, tablename=table_name, data=df)

        return jsonify(
            {
                "status": "success",
                "message": "Data uploaded successfully",
                "rows": len(df),
                "job_id": job_id,
            }
        )
    else:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Invalid file format. Please upload a CSV file.",
                }
            ),
            400,
        )


def process_uploaded_data():
    """Process uploaded data based on user configurations"""

    engine = get_db_engine()
    # Read the JSON input from the frontend
    data = request.get_json()

    job_id = data.get("job_id")
    target_column = data.get("target_column")
    handling_missing = data.get("handling_missing")
    scaling = data.get("scaling")
    encoding = data.get("encoding")

    if not all([job_id, target_column]):
        return jsonify({"status": "error", "message": "Missing required fields."}), 400

    # Load raw data from database
    config = load_config()
    raw_table_name = config["data"]["streamlit"]["unprocessed_data"]
    df = pull_data_from_db(engine, raw_table_name, job_id=job_id)

    if df.empty:
        return (
            jsonify(
                {"status": "error", "message": "No data found for the provided Job ID."}
            ),
            404,
        )

    # Process the dataframe according to options
    processed_df = process_streamlit_dataframe(
        df,
        target_column=target_column,
        handling_missing=handling_missing,
        scaling=scaling,
        encoding=encoding,
    )

    # Push processed data to database
    processed_table_name = config["data"]["streamlit"]["processed_data"]
    push_data_to_db(engine, tablename=processed_table_name, data=processed_df)

    # Return a small preview back to the frontend
    preview = processed_df.head(10).to_json(orient="records")

    return jsonify(
        {
            "status": "success",
            "message": "Data processed and saved successfully",
            "data_preview": preview,
        }
    )


def get_data():
    """Retrieve data from a specified table"""

    engine = get_db_engine()
    data = request.get_json()
    table_name = data["table"]
    job_id = data["job_id"]

    if not table_name:
        return (
            jsonify({"status": "error", "message": "Table name is required"}),
            400,
        )

    # Pull data from database
    df = pull_data_from_db(engine, tablename=table_name, job_id=job_id)

    if df is None or df.empty:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"No data found in table {table_name}",
                }
            ),
            404,
        )

    # Convert DataFrame to JSON and return
    return jsonify(
        {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "rows": len(df),
        }
    )


def ingest(engine):
    """Ingest raw data to database"""
    config = load_config()
    data = config["data"]
    data_path = data["raw_data"]["path"]
    table_name = data["raw_data"]["name"]
    logging.info("Ingesting data to database")
    push_data_to_db(engine, tablename=table_name, dfpath=data_path)
    logging.info("Ingestion Successful")


def process(engine):
    """Process raw data and store in processed table"""
    config = load_config()
    data = config["data"]
    raw_table_name = data["raw_data"]["name"]
    processed_table_name = data["processed_data"]["name"]
    drop_columns = data["processed_data"]["dropcols"]
    target_column = data["processed_data"]["targetcolumn"]
    input_data = pull_data_from_db(engine, tablename=raw_table_name)
    processed_data = process_dataframe(
        input_data, target_column, drop_cols=drop_columns
    )
    push_data_to_db(engine, tablename=processed_table_name, data=processed_data)
