import streamlit as st
import requests
import pandas as pd
import json
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the app
st.set_page_config(page_title="AutoML Dashboard", page_icon="🤖", layout="wide")

# Flask backend URL
FLASK_URL = "http://localhost:8000"  # Change this to match your Flask server

# Session state to track progress
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model_deployed" not in st.session_state:
    st.session_state.model_deployed = False
if "training_metrics" not in st.session_state:
    st.session_state.training_metrics = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "dataset_preview" not in st.session_state:
    st.session_state.dataset_preview = None
if "processed_preview" not in st.session_state:
    st.session_state.processed_preview = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None

# Helper function to make API calls
def make_api_call(endpoint, method="GET", data=None, files=None):
    try:
        url = endpoint  # f"{FLASK_URL}/{endpoint}"

        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data)

        if response.status_code in [200, 201]:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Exception: {str(e)}"


# Main title
st.title("🤖 AutoML System Dashboard")
st.subheader("Automated Machine Learning Pipeline")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "1. Data Ingestion",
    "2. Data Processing",
    "3. Model Training",
    "4. Model Deployment",
    "5. Model Inference",
]
selected_page = st.sidebar.radio("Go to", pages, index=st.session_state.current_step)

# Set the current step based on selection
st.session_state.current_step = pages.index(selected_page)

# Progress bar
st.sidebar.progress((st.session_state.current_step) / (len(pages) - 1))

# Status indicators
status_col1, status_col2, status_col3, status_col4 = st.sidebar.columns(4)
status_col1.metric("Data", "✓" if st.session_state.dataset_uploaded else "✗")
status_col2.metric("Process", "✓" if st.session_state.data_processed else "✗")
status_col3.metric("Train", "✓" if st.session_state.model_trained else "✗")
status_col4.metric("Deploy", "✓" if st.session_state.model_deployed else "✗")

# 1. DATA INGESTION PAGE
if selected_page == "1. Data Ingestion":
    st.header("Data Ingestion")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Preview the data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file_data = uploaded_file
            st.session_state.dataset_preview = df.head(5)
            st.write("Data Preview:")
            st.dataframe(st.session_state.dataset_preview)

            # Show basic stats
            st.write("Basic Statistics:")
            st.dataframe(df.describe())

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Upload Dataset"):
                    # Reset the file position to the beginning
                    uploaded_file.seek(0)

                    # Send to Flask backend
                    files = {"file": uploaded_file}
                    response, error = make_api_call(
                        f"{FLASK_URL}/upload", method="POST", files=files
                    )

                    if error:
                        st.error(error)
                    else:
                        st.session_state.dataset_uploaded = True
                        st.session_state.job_id = response.get("job_id", "unknown")
                        st.success(
                            f"Dataset uploaded successfully! Job ID: {st.session_state.job_id}"
                        )

                        # Get target column options
                        st.session_state.target_column = st.selectbox(
                            "Select target column for modeling", options=df.columns
                        )

            with col2:
                if st.session_state.dataset_uploaded:
                    st.info(
                        f"Dataset ready for processing. Selected target: {st.session_state.target_column}"
                    )
                    if st.button("Proceed to Data Processing"):
                        st.session_state.current_step = 1
                        st.rerun()
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.warning("Please ensure your file is a valid CSV and try again.")
            # Optionally reset the upload state
            if "dataset_preview" in st.session_state:
                del st.session_state.dataset_preview
            if "dataset_uploaded" in st.session_state:
                st.session_state.dataset_uploaded = False

# 2. DATA PROCESSING PAGE
elif selected_page == "2. Data Processing":
    st.header("Data Processing")

    if not st.session_state.dataset_uploaded:
        st.warning("Please upload a dataset first.")
        if st.button("Go to Data Ingestion"):
            st.session_state.current_step = 0
            st.rerun()
    else:
        # Show data preview
        if st.session_state.dataset_preview is not None:
            st.subheader("Original Data Preview")
            st.dataframe(st.session_state.dataset_preview)

        # Processing options
        st.subheader("Processing Options")

        col1, col2 = st.columns(2)

        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Values", ["drop", "mean", "median", "mode", "constant"]
            )

            scaling_method = st.selectbox(
                "Feature Scaling", ["none", "standard", "minmax", "robust"]
            )

        with col2:
            encoding_method = st.selectbox(
                "Categorical Encoding", ["none", "onehot", "label", "target"]
            )

        # Process button
        if st.button("Process Data"):
            # Prepare processing parameters
            processing_params = {
                "job_id": st.session_state.job_id,
                "target_column": st.session_state.target_column,
                "handling_missing": handle_missing,
                "scaling": scaling_method,
                "encoding": encoding_method,
                # "data": st.session_state.uploaded_file_data
            }

            # Call processing endpoint
            with st.spinner("Processing data..."):
                response, error = make_api_call(
                    f"{FLASK_URL}/process_uploaded",
                    method="POST",
                    data=processing_params,
                )

                if error:
                    st.error(error)
                else:
                    st.session_state.data_processed = True
                    st.session_state.processed_preview = response.get(
                        "data_preview", None
                    )

                    # Convert string representation to dataframe if needed
                    if isinstance(st.session_state.processed_preview, str):
                        try:
                            st.session_state.processed_preview = pd.read_json(
                                st.session_state.processed_preview
                            )
                        except:
                            pass

                    st.success("Data processed successfully!")

                    # Show processed data preview
                    if st.session_state.processed_preview is not None:
                        st.subheader("Processed Data Preview")
                        st.dataframe(pd.DataFrame(st.session_state.processed_preview))

                    # Option to proceed
                    if st.button("Proceed to Model Training"):
                        st.session_state.current_step = 2
                        st.rerun()

# 3. MODEL TRAINING PAGE
elif selected_page == "3. Model Training":
    st.header("Model Training")

    if not st.session_state.data_processed:
        st.warning("Please process your data first.")
        if st.button("Go to Data Processing"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)

        with col1:
            problem_type = st.selectbox(
                "Problem Type", ["classification", "regression"]
            )

            models_to_try = st.multiselect(
                "Models to Try",
                ["random_forest", "xgboost", "linear"],
                default=["random_forest", "xgboost"],
            )

        # Train button
        if st.button("Train Models"):
            # Prepare training parameters
            training_params = {
                "job_id": st.session_state.job_id,
                "problem_type": problem_type,
                "models": models_to_try,
            }
            # Call training endpoint
            with st.spinner("Training models... This may take a few minutes"):
                response, error = make_api_call(
                    "http://127.0.0.1:8001/train"
                )  # data=training_params)

                if error:
                    st.error(error)
                else:
                    st.session_state.model_trained = True
                    st.session_state.training_metrics = response.get("metrics", {})
                    st.session_state.feature_importance = response.get(
                        "feature_importance", {}
                    )

                    st.success("Models trained successfully!")

                    # Display training results
                    st.subheader("Training Results")

                    # Display metrics
                    if st.session_state.training_metrics:
                        st.write("Model Performance:")
                        metrics_df = pd.DataFrame(st.session_state.training_metrics)
                        st.dataframe(metrics_df)

                        # # Plot metrics
                        # fig, ax = plt.subplots(figsize=(10, 6))
                        # metrics_df = pd.DataFrame(st.session_state.training_metrics).T
                        # metrics_df.plot(kind='bar', y=metric, ax=ax)
                        # plt.title(f"Model Performance ({metric})")
                        # plt.ylabel(metric)
                        # plt.xticks(rotation=45)
                        # st.pyplot(fig)

                    # Display feature importance
                    if st.session_state.feature_importance:
                        st.write("Feature Importance:")
                        feature_imp = pd.DataFrame(
                            st.session_state.feature_importance, index=[0]
                        ).T
                        feature_imp.columns = ["importance"]
                        feature_imp = feature_imp.sort_values(
                            "importance", ascending=False
                        )

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            x=feature_imp.index, y="importance", data=feature_imp
                        )
                        plt.title("Feature Importance")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)

                    # Option to proceed
                    if st.button("Proceed to Model Deployment"):
                        st.session_state.current_step = 3
                        st.rerun()

# 4. MODEL DEPLOYMENT PAGE
elif selected_page == "4. Model Deployment":
    st.header("Model Deployment")

    if not st.session_state.model_trained:
        st.warning("Please train your model first.")
        if st.button("Go to Model Training"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.subheader("Deployment Configuration")
        deployment_name = st.text_input(
            "Deployment Name", f"automl_model_{st.session_state.job_id}"
        )

        col1, col2 = st.columns(2)

        with col1:
            deployment_type = st.selectbox(
                "Deployment Type", ["rest_api", "batch_inference", "streaming"]
            )

            model_version = st.text_input("Model Version", "v1")

        # Deploy button
        if st.button("Deploy Model"):
            # Prepare deployment parameters
            deployment_params = {
                "job_id": st.session_state.job_id,
                "deployment_name": deployment_name,
                "deployment_type": deployment_type,
                "model_version": model_version,
            }

            # Call deployment endpoint
            with st.spinner("Deploying model..."):
                response, error = make_api_call(
                    "http://127.0.0.1:8002/deploy-auto"
                )  # , method="POST", data=deployment_params

                if error:
                    st.error(error)
                else:
                    st.session_state.model_deployed = True
                    deployment_details = response.get("deployment_details", {})

                    st.success("Model deployed successfully!")

                    # Display deployment details
                    st.subheader("Deployment Details")
                    st.json(deployment_details)

                    # Display endpoint URL if applicable
                    if (
                        deployment_type == "rest_api"
                        and "endpoint_url" in deployment_details
                    ):
                        st.code(f"Endpoint URL: {deployment_details['endpoint_url']}")
                        st.info(
                            "Use this endpoint for making predictions with your model."
                        )

                    # Option to proceed
                    if st.button("Proceed to Model Inference"):
                        st.session_state.current_step = 4
                        st.rerun()

# 5. MODEL INFERENCE PAGE
elif selected_page == "5. Model Inference":
    st.header("Model Inference")

    if not st.session_state.model_deployed:
        st.warning("Please deploy your model first.")
        if st.button("Go to Model Deployment"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.subheader("Make Predictions")

        # Inference options
        inference_method = st.radio(
            "Inference Method", ["Single Prediction", "Batch Prediction"]
        )

        if inference_method == "Single Prediction":
            # Create input form based on original features
            st.write("Enter values for prediction:")

            # In a real app, you would dynamically create input fields based on your feature set
            # For simplicity, we'll create a JSON input area
            input_data = st.text_area(
                "Input JSON (feature values)",
                value='{"feature1": 0.5, "feature2": "category_a", "feature3": 42}',
            )

            if st.button("Predict"):
                try:
                    # Parse input data
                    prediction_data = {
                        "job_id": st.session_state.job_id,
                        "data": json.loads(input_data),
                    }

                    # Call prediction endpoint
                    with st.spinner("Making prediction..."):
                        response, error = make_api_call(
                            "predict", method="POST", data=prediction_data
                        )

                        if error:
                            st.error(error)
                        else:
                            st.success("Prediction complete!")

                            # Display prediction result
                            st.subheader("Prediction Result")
                            st.json(response)
                except Exception as e:
                    st.error(f"Error parsing input data: {str(e)}")

        else:  # Batch Prediction
            uploaded_file = st.file_uploader("Upload batch data (CSV)", type=["csv"])

            if uploaded_file is not None:
                # Preview the data
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write("Batch Data Preview:")
                    st.dataframe(batch_df.head(5))

                    if st.button("Run Batch Prediction"):
                        # Reset the file position to the beginning
                        uploaded_file.seek(0)

                        # # Send to Flask backend
                        # files = {'file': uploaded_file}
                        # data = {'job_id': st.session_state.job_id}

                        csv_buffer = StringIO()
                        batch_df.to_csv(csv_buffer, index=False)
                        csv_string = csv_buffer.getvalue()

                        # Send to Flask backend
                        data = {
                            "job_id": st.session_state.job_id,
                            "csv_data": csv_string,  # Send the CSV data as a string
                        }

                        with st.spinner("Processing batch predictions..."):
                            response, error = make_api_call(
                                "http://127.0.0.1:8002/batch_predict",
                                method="POST",
                                data=data,
                            )

                            if error:
                                st.error(error)
                            else:
                                st.success(
                                    "Batch prediction complete!\n Successfully saved to database"
                                )

                                # Display prediction results
                                st.subheader("Batch Prediction Results")

                                # Convert predictions to dataframe
                                if "predictions" in response:
                                    preds_df = pd.DataFrame(response["predictions"])
                                    st.dataframe(preds_df)

                                    # Download button for predictions
                                    csv = preds_df.to_csv(index=False)
                                    st.download_button(
                                        "Download Predictions",
                                        csv,
                                        "predictions.csv",
                                        "text/csv",
                                        key="download-csv",
                                    )
                except Exception as e:
                    st.error(f"Error reading the batch file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 AutoML System | Made with Streamlit")

# Include refresh button at bottom right
if st.button("Refresh Dashboard"):
    st.rerun()
