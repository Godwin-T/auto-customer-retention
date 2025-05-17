import streamlit as st
from utils.api import make_api_call
from config import ENDPOINTS


def show_model_training():
    """Display the Model Training page"""
    st.header("Model Training")

    if not st.session_state.get("data_processed", False):
        st.warning("Please process your data first.")
        if st.button("Go to Data Processing"):
            st.session_state["current_step"] = 1
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
                    ENDPOINTS["train"], data=training_params, method="POST"
                )

                if error:
                    st.error(error)
                else:
                    st.session_state["model_trained"] = True
                    st.session_state["training_metrics"] = response.get("metrics", {})
                    st.session_state["feature_importance"] = response.get(
                        "feature_importance", {}
                    )

                    st.success("Models trained successfully!")

        # Make the navigation button visible only when models have been trained
        # Keeping it outside the train button block to be consistent with your working example
        if st.session_state.get("model_trained", False):
            if st.button("Proceed to Model Deployment", key="proceed_to_deployment"):
                st.session_state["current_step"] = 3
                st.rerun()
