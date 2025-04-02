# ML Monitoring with Prefect

This repository contains a comprehensive ML monitoring solution for a customer retention prediction system. It uses Prefect for workflow orchestration and includes data drift detection, model performance tracking, and system metrics monitoring.

## Architecture

The monitoring system consists of the following components:

1. **Monitoring Service**: Collects metrics, detects data drift, and evaluates model performance
2. **Prefect Service**: Orchestrates monitoring workflows and retraining pipelines
3. **Prometheus**: Stores time-series metrics data
4. **MLflow**: Tracks experiments and models

## Setup

To set up the monitoring infrastructure, run:

```bash
# Clone the repository
git clone https://github.com/your-username/ml-monitoring.git
cd ml-monitoring

# Run the setup script
chmod +x setup_monitoring.sh
./setup_monitoring.sh
```

## Components

### Monitoring Service

The monitoring service is responsible for:
- Collecting model performance metrics
- Detecting data drift
- Monitoring system resources
- Providing an API for accessing monitoring data

### Prefect Flows

The repository includes the following Prefect flows:
- **Monitoring Flow**: Runs hourly to check for data drift, model performance, and system health
- **Retraining Flow**: Runs daily to check if retraining is needed and starts a retraining job if necessary

### Configuration

All monitoring settings are defined in `configs/monitoring/config.yaml`. You can customize:
- Models to monitor
- Metrics thresholds
- Drift detection parameters
- Alert settings

## Accessing Monitoring Data

- **Monitoring API**: Available at http://localhost:8003
- **Prefect Dashboard**: Available at http://localhost:4200
- **Prometheus Metrics**: Available at http://localhost:9090

## Adding a New Model to Monitor

To add a new model to the monitoring system:

1. Update `configs/monitoring/config.yaml` to include the new model:
   ```yaml
   models:
     - name: "new_model_name"
       metrics:
         - accuracy
         - precision
         - recall
       thresholds:
         accuracy: 0.80
   ```

2. Restart the monitoring service:
   ```bash
   docker-compose restart monitoring
   ```

## Alerts

The system sends alerts when:
- Data drift is detected
- Model performance drops below thresholds
- System resources are constrained

Alerts are sent via:
- Email
- Slack (configurable webhook)

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   docker-compose logs monitoring
   docker-compose logs prefect
   ```

2. Verify the configurations in `configs/monitoring/config.yaml`

3. Ensure all services are running:
   ```bash
   docker-compose ps
   ```
