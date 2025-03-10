# FROM python:3.9.7-slim

# # Install required system packages
# RUN apt-get update && apt-get install -y \
#     gcc \
#     libsqlite3-dev \
#    # python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip
# RUN pip install -U pip

# # Set working directory
# WORKDIR /home

# # Copy requirements file
# COPY requirements.txt .

# # Install dependencies

# RUN --mount=type=cache,target=/root/.cache/pip/http \
#     pip install -r requirements.txt

#     # Copy application files
# COPY . /home/

# # Set entrypoint
# CMD ["python", "ingestion.py"]


FROM python:3.9-alpine as builder

WORKDIR /app

# Alpine needs these dependencies for scientific packages
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev \
    sqlite-dev \
    linux-headers

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.9-alpine

WORKDIR /app

# Install only the required runtime libraries
RUN apk add --no-cache sqlite-libs libstdc++

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# # Copy application code
COPY . .

CMD ["python", "ingestion.py"]
