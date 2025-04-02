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
