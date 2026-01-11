#!/bin/bash
# Start both Streamlit app and FastAPI webhook service locally
# Usage: ./start_local.sh

set -e

STREAMLIT_PORT=8502
FASTAPI_PORT=8010
FASTAPI_HOST="127.0.0.1"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $FASTAPI_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
    echo "✓ All services stopped"
    exit 0
}

# Register cleanup on exit
trap cleanup SIGINT SIGTERM EXIT

echo "========================================================================"
echo "TicketsAnalyzerAI - Local Development Server"
echo "========================================================================"

# Load .env file if present
if [ -f ".env" ]; then
    echo "📋 Loading environment from .env..."
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
    echo "✓ Environment variables loaded"
fi

# Start FastAPI (optional)
if [ -f "service_webhook.py" ]; then
    echo "🚀 Starting FastAPI webhook service on $FASTAPI_HOST:$FASTAPI_PORT..."
    python -m uvicorn service_webhook:app \
        --host "$FASTAPI_HOST" \
        --port "$FASTAPI_PORT" \
        --log-level info &
    FASTAPI_PID=$!
    echo "✓ FastAPI started (PID: $FASTAPI_PID)"
    sleep 2
else
    echo "⚠️  service_webhook.py not found, skipping webhook service"
fi

# Start Streamlit (required)
echo "🚀 Starting Streamlit app on port $STREAMLIT_PORT..."
python -m streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address localhost &
STREAMLIT_PID=$!
echo "✓ Streamlit started (PID: $STREAMLIT_PID)"

echo ""
echo "========================================================================"
echo "✓ All services started successfully!"
echo "========================================================================"
echo "📱 Streamlit UI:    http://localhost:$STREAMLIT_PORT"
if [ -n "$FASTAPI_PID" ]; then
    echo "🔌 FastAPI Webhook: http://$FASTAPI_HOST:$FASTAPI_PORT"
    echo "📖 API Docs:        http://$FASTAPI_HOST:$FASTAPI_PORT/docs"
fi
echo "========================================================================"
echo "Press Ctrl+C to stop all services"
echo "========================================================================"

# Wait for processes
wait
