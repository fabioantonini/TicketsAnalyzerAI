#!/bin/bash
# Docker entrypoint - starts both Streamlit and FastAPI services
# This script is used inside the Docker container

set -e

echo "========================================================================"
echo "TicketsAnalyzerAI - Docker Multi-Service Startup"
echo "========================================================================"

# Start FastAPI in background (if service_webhook.py exists)
if [ -f "service_webhook.py" ]; then
    echo "🚀 Starting FastAPI webhook service on 0.0.0.0:8010..."
    python -m uvicorn service_webhook:app \
        --host 0.0.0.0 \
        --port 8010 \
        --log-level info &
    FASTAPI_PID=$!
    echo "✓ FastAPI started (PID: $FASTAPI_PID)"
else
    echo "⚠️  service_webhook.py not found, skipping webhook service"
fi

# Start Streamlit in foreground (main process)
echo "🚀 Starting Streamlit app on 0.0.0.0:8501..."
echo "========================================================================"
exec streamlit run app.py
