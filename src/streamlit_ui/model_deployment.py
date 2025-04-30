# import streamlit as st
# from utils.api import make_api_call
# from config import ENDPOINTS

# def show_model_deployment():
#     """Display the Model Deployment page"""
#     st.header("Model Deployment")

#     if not st.session_state.model_trained:
#         st.warning("Please train your model first.")
#         if st.button("Go to Model Training"):
#             st.session_state.current_step = 2
#             st.rerun()
#     else:
#         st.subheader("Deployment Configuration")
#         deployment_name = st.text_input(
#             "Deployment Name", f"automl_model_{st.session_state.job_id}"
#         )

#         col1, col2 = st.columns(2)

#         with col1:
#             deployment_type = st.selectbox(
#                 "Deployment Type", ["rest_api", "batch_inference", "streaming"]
#             )

#             model_version = st.text_input("Model Version", "v1")

#         # Deploy button
#         if st.button("Deploy Model"):
#             # Prepare deployment parameters
#             deployment_params = {
#                 "job_id": st.session_state.job_id,
#                 "deployment_name": deployment_name,
#                 "deployment_type": deployment_type,
#                 "model_version": model_version,
#             }

#             # Call deployment endpoint
#         with st.spinner("Deploying model..."):
#             response, error = make_api_call(ENDPOINTS["deploy"])  # Add deployment_params when API is ready
#             if error:
#                 st.error(error)
#             else:
#                 # Check if deployment was successful based on the 'deployed' field
#                 if response.get("deployed", False):
#                     st.session_state.model_deployed = True
#                     st.success(f"✅ {response.get('response', 'Model deployed successfully!')}")

#                     # Display deployment details
#                     st.subheader("Deployment Details")

#                     # Create columns for key metrics display
#                     col1, col2 = st.columns(2)

#                     with col1:
#                         st.metric("Model Version", response.get("model_version", "N/A"))
#                         st.write("**Status:**", response.get("status", "Completed"))

#                     with col2:
#                         # Display key metrics if available
#                         metrics = response.get("metrics", {})
#                         if metrics:
#                             st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
#                             st.metric("Accuracy", f"{metrics.get('accuracy_score', 0):.4f}")

#                     # Display full response in expandable section
#                     with st.expander("View Full Deployment Response"):
#                         st.json(response)

#                     # Display endpoint URL if applicable
#                     if deployment_type == "rest_api" and "endpoint_url" in response:
#                         st.code(f"Endpoint URL: {response['endpoint_url']}")
#                         st.info("Use this endpoint for making predictions with your model.")

#                     # Option to proceed
#                     if st.button("Proceed to Model Inference"):
#                         st.session_state.current_step = 4
#                         st.rerun()
#                 else:
#                     # Model was not deployed - show reason
#                     st.warning(f"⚠️ {response.get('response', 'Model was not deployed')}")

#                     # Display reason if available
#                     if "reason" in response:
#                         st.info(f"Reason: {response['reason']}")
#                     elif "error" in response:
#                         st.error(f"Error: {response['error']}")

#                     # Display status
#                     st.write("**Status:**", response.get("status", "Failed"))

#                     # Show metrics if available even though not deployed
#                     metrics = response.get("metrics", {})
#                     if metrics:
#                         st.subheader("Model Metrics")
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
#                         with col2:
#                             st.metric("Accuracy", f"{metrics.get('accuracy_score', 0):.4f}")

#                     # Display full response in expandable section
#                     with st.expander("View Full Deployment Response"):
#                         st.json(response)

#                     # Option to retry or proceed
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if st.button("Retry Deployment"):
#                             st.rerun()
#                     with col2:
#                         if st.button("Proceed Anyway"):
#                             st.session_state.current_step = 4
#                             st.rerun()


import streamlit as st
from utils.api import make_api_call
from config import ENDPOINTS


def show_model_deployment():
    """Display the Model Deployment page"""
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

            # Call deployment endpoint - FIX: proper indentation here
            with st.spinner("Deploying model..."):
                response, error = make_api_call(
                    ENDPOINTS["deploy"], data=deployment_params
                )  # Added deployment_params
                if error:
                    st.error(error)
                else:
                    # Check if deployment was successful based on the 'deployed' field
                    if response.get("deployed", False):
                        st.session_state.model_deployed = True
                        st.success(
                            f"✅ {response.get('response', 'Model deployed successfully!')}"
                        )

                        # Display deployment details
                        st.subheader("Deployment Details")

                        # Create columns for key metrics display
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Model Version", response.get("model_version", "N/A")
                            )
                            st.write("**Status:**", response.get("status", "Completed"))

                        with col2:
                            # Display key metrics if available
                            metrics = response.get("metrics", {})
                            if metrics:
                                st.metric(
                                    "F1 Score", f"{metrics.get('f1_score', 0):.4f}"
                                )
                                st.metric(
                                    "Accuracy",
                                    f"{metrics.get('accuracy_score', 0):.4f}",
                                )

                        # Display full response in expandable section
                        with st.expander("View Full Deployment Response"):
                            st.json(response)

                        # Display endpoint URL if applicable
                        if deployment_type == "rest_api" and "endpoint_url" in response:
                            st.code(f"Endpoint URL: {response['endpoint_url']}")
                            st.info(
                                "Use this endpoint for making predictions with your model."
                            )
                    else:
                        # Model was not deployed - show reason
                        st.warning(
                            f"⚠️ {response.get('response', 'Model was not deployed')}"
                        )

                        # Display reason if available
                        if "reason" in response:
                            st.info(f"Reason: {response['reason']}")
                        elif "error" in response:
                            st.error(f"Error: {response['error']}")

                        # Display status
                        st.write("**Status:**", response.get("status", "Failed"))

                        # Show metrics if available even though not deployed
                        metrics = response.get("metrics", {})
                        if metrics:
                            st.subheader("Model Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "F1 Score", f"{metrics.get('f1_score', 0):.4f}"
                                )
                            with col2:
                                st.metric(
                                    "Accuracy",
                                    f"{metrics.get('accuracy_score', 0):.4f}",
                                )

                        # Display full response in expandable section
                        with st.expander("View Full Deployment Response"):
                            st.json(response)

                        # Option to retry or proceed
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Retry Deployment"):
                                st.rerun()
                        with col2:
                            if st.button("Proceed Anyway"):
                                st.session_state.current_step = 4
                                st.rerun()

        # Move the navigation button outside the deploy logic
        # This will be visible once a model is deployed
        if st.session_state.get("model_deployed", False):
            st.write("---")
            st.write("Ready to test your deployed model?")
            if st.button("Proceed to Model Inference", key="proceed_to_inference"):
                st.session_state.current_step = 4
                st.rerun()
