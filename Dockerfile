# Build Stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime Stage
FROM python:3.12-slim

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y \
    openssh-client \
    curl \
    netcat-traditional \
    net-tools \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Create SSH directory and set permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Copy the Python application
COPY api.py .
COPY configs/ /app/configs/
COPY langgraph /app/langgraph/
COPY smolagents /app/smolagents/

# Copy dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/

# Copy SSH tunnel script
COPY scripts/tunnel.sh /app/tunnel.sh
RUN chmod +x /app/tunnel.sh

# Copy the SSH key (only needed for local development)
COPY ssh/sandbox_key.pem /root/.ssh/
RUN chmod 600 /root/.ssh/sandbox_key.pem

# Expose the application's port
EXPOSE 5000

# Use the tunnel script as entrypoint
ENTRYPOINT ["/app/tunnel.sh"]