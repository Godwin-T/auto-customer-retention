import streamlit as st


def initialize_session_state():
    """Initialize all session state variables if they don't exist"""
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
    if "uploaded_file_data" not in st.session_state:
        st.session_state.uploaded_file_data = None
