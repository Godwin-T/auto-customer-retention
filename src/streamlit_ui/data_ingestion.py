import streamlit as st
import pandas as pd
from utils.api import make_api_call
from config import ENDPOINTS


def show_data_ingestion():
    """Display the Data Ingestion page"""
    st.header("Data Ingestion")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file_data = uploaded_file
            st.session_state.dataset_preview = df.head(5)
            st.write("Data Preview:")
            st.dataframe(st.session_state.dataset_preview)

            st.write("Basic Statistics:")
            st.dataframe(df.describe())

            if st.button("Upload Dataset"):
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                response, error = make_api_call(
                    ENDPOINTS["upload"], method="POST", files=files
                )

                if error:
                    st.error(error)
                else:
                    st.session_state.dataset_uploaded = True
                    st.session_state.job_id = response.get("job_id", "unknown")
                    st.success(
                        f"Dataset uploaded successfully! Job ID: {st.session_state.job_id}"
                    )

        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.warning("Please ensure your file is a valid CSV and try again.")
            if "dataset_preview" in st.session_state:
                del st.session_state.dataset_preview
            if "dataset_uploaded" in st.session_state:
                st.session_state.dataset_uploaded = False

    # 🔥 Now handle this separately: AFTER dataset uploaded
    if st.session_state.get("dataset_uploaded", False):

        if st.button("Proceed to Data Processing"):
            st.session_state.current_step = 1
            st.rerun()
