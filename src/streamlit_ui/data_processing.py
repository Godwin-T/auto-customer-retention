import streamlit as st
import pandas as pd
from utils.api import make_api_call
from config import ENDPOINTS


def show_data_processing():
    """Display the Data Processing page"""
    st.header("Data Processing")

    if not st.session_state.get("dataset_uploaded", False):
        st.warning("Please upload a dataset first.")
        if st.button("Go to Data Ingestion"):
            st.session_state.current_step = 0
            st.rerun()
    else:
        # Show data preview
        if st.session_state.get("dataset_preview") is not None:
            st.subheader("Original Data Preview")
            st.dataframe(st.session_state.dataset_preview)

        # 🔥 New: Let user pick the target column
        df = st.session_state.dataset_preview  # reload preview
        st.subheader("Select Target Column for Modeling")
        target_column = st.selectbox("Target Column", options=df.columns)

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
                "target_column": target_column,
                "handling_missing": handle_missing,
                "scaling": scaling_method,
                "encoding": encoding_method,
            }

            # Call processing endpoint
            with st.spinner("Processing data..."):
                response, error = make_api_call(
                    ENDPOINTS["process"], method="POST", data=processing_params
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
        if st.session_state.get("processed_preview") is not None:
            st.subheader("Processed Data Preview")
            st.dataframe(pd.DataFrame(st.session_state.processed_preview))

        # Make the navigation button visible only when data has been processed
        if st.session_state.get("data_processed", False):
            # Using st.rerun() instead of st.experimental_rerun()
            if st.button("Proceed to Model Training", key="proceed_to_training"):
                st.session_state["current_step"] = 2
                st.rerun()
