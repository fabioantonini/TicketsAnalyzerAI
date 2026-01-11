# Dockerfile for Streamlit RAG Support App

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py ./
COPY service_webhook.py ./
COPY rag_core.py ./
COPY mcp_prompts.json ./
COPY README.md ./
COPY start_docker.sh ./

# Make startup script executable
RUN chmod +x start_docker.sh

# Expose ports (Streamlit + FastAPI)
EXPOSE 8501 8010

# Streamlit configuration to allow access from container
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Default command (starts both Streamlit and FastAPI)
CMD ["./start_docker.sh"]
