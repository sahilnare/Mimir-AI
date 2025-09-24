#!/bin/sh

# Exit on any error
set -e

# Check if APP_ENV is set
if [ -z "${APP_ENV}" ]; then
    echo "No APP_ENV detected - running in local development mode with SSH tunneling..."
    
    # Start SSH tunnel in background with your local configuration
    echo "Setting up SSH tunnel to PostgreSQL RDS..."
    
    # Use sandbox configuration for local development
    ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i /root/.ssh/sandbox_key.pem \
        -f -N -L 63333:db-sandbox-for-ai.cyuh3uofyiy4.ap-south-1.rds.amazonaws.com:5432 \
        ubuntu@ec2-13-127-111-83.ap-south-1.compute.amazonaws.com

    # Wait for tunnel to establish
    sleep 2

    # Verify tunnel is established
    if ! netstat -an | grep "LISTEN" | grep -q ":63333"; then
        echo "Failed to establish SSH tunnel on port 63333"
        exit 1
    fi

    echo "SSH tunnel successfully established on port 63333"
    echo "Database accessible at localhost:63333"

else
    echo "Running in ${APP_ENV} environment - no SSH tunnel needed..."
    echo "Application will use direct RDS connection via ECS/EC2 network"
fi

# Start the main application
echo "Starting application..."
exec uvicorn api:app --host 0.0.0.0 --port 5000