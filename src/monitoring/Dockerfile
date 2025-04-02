FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for Prometheus and the monitoring API
EXPOSE 8003 8005

# Command to run the monitoring service
CMD ["python", "app.py"]
