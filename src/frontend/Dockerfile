# Use the official Python image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "automl_dashboard.py"]
