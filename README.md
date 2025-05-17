# Customer Retention System

A comprehensive machine learning system that handles the entire ML lifecycle from data ingestion to model deployment, focused on predicting and improving customer retention.

## Project Overview

This project implements a complete machine learning pipeline for customer retention prediction with:

- Data ingestion and processing
- Model training and hyperparameter tuning
- Model evaluation and comparison
- Model deployment as microservices
- Model monitoring and performance tracking
- User-friendly interfaces for interacting with the system

The system is designed with a microservices architecture using Docker containers, ensuring scalability and ease of deployment.

## Architecture

```
customer-retention-system/
├── .github/workflows/      # CI pipeline configuration
├── assets/                 # Project images and resources
├── configs/                # Configuration management
├── data/                   # Input data storage
├── databases/              # Database scripts and migrations
├── docker-compose.yml      # Container orchestration
├── example-env.sh          # Example .env
├── migrations/             # Database migrations
├── notebooks/              # Experimental Jupyter notebooks
├── prepare_db.sh           # Database preparation script
├── requirements.txt        # Dependencies
├── src/
│   ├── core/               # Core functionality
│   │   ├── common/         # Shared utilities
│   │   ├── ingestion/      # Data ingestion and processing
│   │   ├── training/       # Model training pipeline
│   │   └── deployment/     # Model deployment services
│   ├── monitoring/         # Monitoring services (Prefect)
│   ├── streamlit_ui/       # Streamlit user interface
│   └── tmp/                # Temporary files
└── tests/                  # Automated tests
```

## Key Features

- **End-to-End ML Pipeline**: Handles all aspects of the machine learning lifecycle
- **Microservices Architecture**: Components are containerized for scalability and isolation
- **Model Versioning & Tracking**: Integrated with MLflow for experiment tracking
- **Workflow Orchestration**: Utilizes prefect for workflow orchestration
- **CI Integration**: Automated testing with GitHub Actions
- **Multiple Interfaces**: Flask API endpoints and Streamlit UI for interacting with the system
- **Extensible Model Framework**: Easily add or swap models as needed

## Models Implemented

The system includes several regression models for customer retention prediction:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- XGBoost Regression

The modular design makes it easy to add new models to the pipeline based on your specific needs.

## Adaptability

While built for customer retention prediction, this framework can be readily adapted for other machine learning use cases by:
- Updating data ingestion components for new data sources
- Modifying preprocessing steps for different data types
- Adding new machine learning models appropriate for the specific problem
- Adjusting evaluation metrics based on the new objectives
- Customizing the UI/API endpoints for the new application

The microservices architecture ensures that individual components can be modified without disrupting the entire system.

## Workflow Orchestration with Prefect

The project uses Prefect for orchestrating the ML pipeline workflows. Below are visualizations of the flows:

### Scheduled Runs
![Prefect Scheduled Runs](./assets/runs.png)

### Flow Diagram
![Prefect Flow Diagram](./assets/flow.png)

## Technologies Used

- **Machine Learning**: scikit-learn, XGBoost, TensorFlow/Keras
- **Workflow Orchestration**: Prefect
- **Model Tracking**: MLflow
- **Containerization**: Docker
- **Database**: SQLite 
- **Web Interfaces**: Flask, Streamlit
- **CI**: GitHub Actions

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.8+

### Installation
1. Clone the repository
   ```
   https://github.com/Godwin-T/auto-customer-retention.git
   cd customer-retention-system
   ```

2. Run the system using Docker Compose
   ```
   chmod +x prepare_db.sh
   ./prepare_db.sh
   docker-compose up
   ```

3. Access the Streamlit UI at http://localhost:8501

## Usage

### API Endpoints
- `/predict` - Get retention predictions for new customers
- `/process` - Process new data
- `/train` - Train models with specified parameters

### Streamlit UI
The Streamlit interface provides an interactive way to:
- Upload customer data
- Process and save data to database
- Configure and train models with custom parameters
- View model performance metrics and comparisons
- Deploy models to production

## Future Improvements

### Database Migration
The current system uses SQLite, which has limitations:
- **Concurrency Issues**: SQLite experiences locking problems with concurrent operations
- **Scalability Constraints**: Not suitable for high-volume production environments

**Planned PostgreSQL Implementation**:
- Better handling of concurrent operations
- Improved performance for larger datasets
- More robust transaction support
- Better integration with monitoring tools

### Model Enhancements
- **Advanced Hyperparameter Tuning**: Implement Grid Search and Random Search for optimal parameter selection
- **AutoML Integration**: Add support for automated model selection and feature engineering
- **Ensemble Methods**: Implement stacking and blending techniques for improved prediction accuracy
- **Custom Loss Functions**: Support for specialized loss functions tailored to retention prediction

### Flexibility Improvements
- **Parameter Customization**: Enhanced configuration options for all pipeline stages
- **Feature Selection Algorithms**: Automatic feature importance analysis and selection
- **Data Drift Detection**: Automated detection of data distribution changes
- **A/B Testing Framework**: Compare model versions in production

## Development

### Adding New Models
1. Create a new model class in `src/core/training/`
2. Register the model in the model registry
3. Update configuration as needed

### Database Migration Guide
To switch from SQLite to PostgreSQL:
1. Update database connection settings in `config.yaml`
2. Run migration script: `python src/core/common/db_migration.py`
3. Update Docker Compose file to include PostgreSQL service
