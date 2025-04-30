import streamlit as st
from utils.session_state import initialize_session_state
from data_ingestion import show_data_ingestion
from data_processing import show_data_processing
from model_training import show_model_training
from model_deployment import show_model_deployment
from model_inference import show_model_inference
from config import APP_TITLE, APP_ICON, PAGES


def main():
    # Configure the app
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    # Initialize session state variables
    initialize_session_state()

    # Main title
    st.title("🤖 AutoML System Dashboard")
    st.subheader("Automated Machine Learning Pipeline")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Go to", PAGES, index=st.session_state.current_step
    )

    # Set the current step based on selection
    st.session_state.current_step = PAGES.index(selected_page)

    # Progress bar
    st.sidebar.progress((st.session_state.current_step) / (len(PAGES) - 1))

    # Status indicators
    status_col1, status_col2, status_col3, status_col4 = st.sidebar.columns(4)
    status_col1.metric("Data", "✓" if st.session_state.dataset_uploaded else "✗")
    status_col2.metric("Process", "✓" if st.session_state.data_processed else "✗")
    status_col3.metric("Train", "✓" if st.session_state.model_trained else "✗")
    status_col4.metric("Deploy", "✓" if st.session_state.model_deployed else "✗")

    # Display the selected page
    if selected_page == "1. Data Ingestion":
        show_data_ingestion()
    elif selected_page == "2. Data Processing":
        show_data_processing()
    elif selected_page == "3. Model Training":
        show_model_training()
    elif selected_page == "4. Model Deployment":
        show_model_deployment()
    elif selected_page == "5. Model Inference":
        show_model_inference()

    # Footer
    st.markdown("---")
    st.markdown("© 2025 AutoML System | Made with Streamlit")

    # Include refresh button at bottom right
    if st.button("Refresh Dashboard"):
        st.rerun()


if __name__ == "__main__":
    main()
