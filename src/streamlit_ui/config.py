# App configuration constants
APP_TITLE = "AutoML Dashboard"
APP_ICON = "🤖"

# Navigation pages
PAGES = [
    "1. Data Ingestion",
    "2. Data Processing",
    "3. Model Training",
    "4. Model Deployment",
    "5. Model Inference",
]

# API endpoints
ENDPOINTS = {
    "upload": f"http://localhost:8002/ingestion/upload",
    "process": f"http://localhost:8002/ingestion/process_uploaded",
    "train": "http://localhost:8002/training/train",  # Using the original URL as in the source
    "deploy": "http://localhost:8002/deployment/deploy-auto",  # Using the original URL as in the source
    "batch_predict": "http://localhost:8002/deployment/batch_predict",  # Using the original URL as in the source
}
