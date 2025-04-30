import streamlit as st
import pandas as pd
import json
from io import StringIO
from utils.api import make_api_call
from config import ENDPOINTS


def show_model_inference():
    """Display the Model Inference page"""
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
                            ENDPOINTS["predict"], method="POST", data=prediction_data
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
                                ENDPOINTS["batch_predict"], method="POST", data=data
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

                                    # Downlod button for predictions
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
