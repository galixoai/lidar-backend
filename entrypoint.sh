#!/bin/bash
set -e

echo "Waiting for Temporal server..."
while ! nc -z temporal-worker 7233; do   
  echo "Waiting for Temporal..."
  sleep 1
done
echo "Temporal server is up."

cd /app

# Start the FastAPI server in the background
echo "Starting FastAPI server..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start the Temporal worker
echo "Starting Temporal Worker..."
python -m app.worker

# Wait for all background processes
wait
