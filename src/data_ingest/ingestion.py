import os
import yaml
import logging
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from utils import push_data_to_db, process_dataframe, pull_data_from_db, connect_sqlite

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def load_config():
    config_path = os.getenv("config_path")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def get_db_engine():
    config = load_config()
    return connect_sqlite(config["database"]["db_path"])


def create_app(engine=None):
    app = Flask("Data Ingestion and Processing")
    if engine is None:
        engine = get_db_engine()

    @app.route("/process", methods=["GET"])
    def process_existing_data():
        """Process data already in the database"""
        try:
            ingest(engine)
            process(engine)
            return jsonify(
                {"status": "success", "message": "Successfully processed existing data"}
            )
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/upload", methods=["POST"])
    def upload_data():
        """Endpoint to receive data uploads from Streamlit"""
        try:
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

                # Get table name from config
                config = load_config()
                table_name = config["data"]["raw_data"]["name"]

                # Push data to database
                push_data_to_db(engine, tablename=table_name, data=df)

                return jsonify(
                    {
                        "status": "success",
                        "message": "Data uploaded successfully",
                        "rows": len(df),
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

        except Exception as e:
            logger.error(f"Error uploading data: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/process_uploaded", methods=["POST"])
    def process_uploaded_data():
        """Process uploaded data directly without saving to raw table first"""
        try:

            process(engine)
            return jsonify(
                {"status": "success", "message": "Successfully processed existing data"}
            )
            # Check if a file was uploaded
            # if 'job_id' not in request.keys():
            #     return jsonify({"status": "error", "message": "No file part"}), 400

            # data = request

            # # Check if file is empty
            # if file.filename == '':
            #     return jsonify({"status": "error", "message": "No selected file"}), 400

            # # Read the file into a pandas DataFrame
            # if file and file.filename.endswith('.csv'):
            #     # Read CSV file into DataFrame
            #     df = pd.read_csv(file)

            #     # Get configuration
            #     config = load_config()
            #     processed_table_name = config["data"]["processed_data"]["name"]
            #     drop_columns = config["data"]["processed_data"]["dropcols"]
            #     target_column = config["data"]["process_data"]["targetcolumn"]

            #     # Process the data
            #     processed_data = process_dataframe(df, target_column, drop_cols=drop_columns)

            #     # Push processed data to database
            #     push_data_to_db(engine, tablename=processed_table_name, data=processed_data)

            #     return jsonify({
            #         "status": "success",
            #         "message": "Data processed and saved successfully",
            #         "rows": len(processed_data)
            #     })
            # else:
            #     return jsonify({"status": "error", "message": "Invalid file format. Please upload a CSV file."}), 400

        except Exception as e:
            logger.error(f"Error processing uploaded data: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/get_data", methods=["GET"])
    def get_data():
        """Retrieve data from a specified table"""
        try:
            table_name = request.args.get("table")
            if not table_name:
                return (
                    jsonify({"status": "error", "message": "Table name is required"}),
                    400,
                )

            # Pull data from database
            df = pull_data_from_db(engine, tablename=table_name)

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

        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return app


def ingest(engine):
    config = load_config()
    data = config["data"]
    data_path = data["raw_data"]["path"]
    table_name = data["raw_data"]["name"]
    logging.info("Ingesting data to database")
    push_data_to_db(engine, tablename=table_name, dfpath=data_path)
    logging.info("Ingestion Successful")


def process(engine):
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


# Only run the app directly if main
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8000, host="0.0.0.0")
