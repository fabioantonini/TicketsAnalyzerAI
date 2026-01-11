#!/usr/bin/env python3
"""
Start both Streamlit app and FastAPI webhook service locally.
Usage: python start_local.py
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

# Configuration
STREAMLIT_PORT = 8502
FASTAPI_PORT = 8010
FASTAPI_HOST = "127.0.0.1"

processes = []


def cleanup(signum=None, frame=None):
    """Terminate all child processes on exit."""
    print("\n🛑 Shutting down services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("✓ All services stopped")
    sys.exit(0)


def start_fastapi():
    """Start FastAPI webhook service."""
    print(f"🚀 Starting FastAPI webhook service on {FASTAPI_HOST}:{FASTAPI_PORT}...")

    # Check if service_webhook.py exists
    if not Path("service_webhook.py").exists():
        print("⚠️  service_webhook.py not found, skipping webhook service")
        return None

    cmd = [
        sys.executable, "-m", "uvicorn",
        "service_webhook:app",
        "--host", FASTAPI_HOST,
        "--port", str(FASTAPI_PORT),
        "--log-level", "info"
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        print(f"✓ FastAPI started (PID: {proc.pid})")
        return proc
    except Exception as e:
        print(f"✗ Failed to start FastAPI: {e}")
        return None


def start_streamlit():
    """Start Streamlit app."""
    print(f"🚀 Starting Streamlit app on port {STREAMLIT_PORT}...")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "localhost"
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        print(f"✓ Streamlit started (PID: {proc.pid})")
        return proc
    except Exception as e:
        print(f"✗ Failed to start Streamlit: {e}")
        return None


def main():
    """Main entry point."""
    print("=" * 80)
    print("TicketsAnalyzerAI - Local Development Server")
    print("=" * 80)

    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start FastAPI (optional)
    fastapi_proc = start_fastapi()
    if fastapi_proc:
        processes.append(fastapi_proc)
        time.sleep(2)  # Give it time to start

    # Start Streamlit (required)
    streamlit_proc = start_streamlit()
    if not streamlit_proc:
        cleanup()
        sys.exit(1)
    processes.append(streamlit_proc)

    print("\n" + "=" * 80)
    print("✓ All services started successfully!")
    print("=" * 80)
    print(f"📱 Streamlit UI:    http://localhost:{STREAMLIT_PORT}")
    if fastapi_proc:
        print(f"🔌 FastAPI Webhook: http://{FASTAPI_HOST}:{FASTAPI_PORT}")
        print(f"📖 API Docs:        http://{FASTAPI_HOST}:{FASTAPI_PORT}/docs")
    print("=" * 80)
    print("Press Ctrl+C to stop all services")
    print("=" * 80)

    # Monitor processes
    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  Process {proc.pid} exited with code {proc.returncode}")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
