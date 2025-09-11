#!/bin/bash

# MLX Whisper Wyoming Server Startup Script

echo "Starting MLX Whisper Wyoming Server..."

# Set environment variables (optional - override defaults)
export STT_HOST=${STT_HOST:-"0.0.0.0"}
export STT_PORT=${STT_PORT:-"10300"}

# Install/update dependencies using uv
echo "Installing dependencies with uv..."
uv sync

# Start the Wyoming server using uv run
echo "Starting Wyoming server on ${STT_HOST}:${STT_PORT}"
uv run python server.py --host "${STT_HOST}" --port "${STT_PORT}"