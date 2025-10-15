#!/bin/bash

# Install MLX Whisper Wyoming Protocol Service for macOS

set -e  # Exit on error

SERVICE_NAME="com.mlx-whisper.wyoming"
PLIST_TEMPLATE="${SERVICE_NAME}.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# Detect paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")

# Verify uv exists
if [ ! -f "$UV_PATH" ]; then
    echo "Error: uv not found. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Verify template exists
if [ ! -f "$SCRIPT_DIR/$PLIST_TEMPLATE" ]; then
    echo "Error: plist template not found: $PLIST_TEMPLATE"
    exit 1
fi

# Detect ffmpeg path (usually in homebrew)
FFMPEG_DIR=$(dirname $(which ffmpeg 2>/dev/null || echo "/opt/homebrew/bin/ffmpeg"))

# Build PATH for the service
SERVICE_PATH="${FFMPEG_DIR}:$(dirname $UV_PATH):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

echo "Installing MLX Whisper Wyoming Protocol Service..."
echo "  Service name: $SERVICE_NAME"
echo "  Working directory: $SCRIPT_DIR"
echo "  uv path: $UV_PATH"
echo "  Service PATH: $SERVICE_PATH"
echo ""

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Process template and install plist
sed -e "s|__UV_PATH__|${UV_PATH}|g" \
    -e "s|__WORKING_DIR__|${SCRIPT_DIR}|g" \
    -e "s|__SERVICE_PATH__|${SERVICE_PATH}|g" \
    "$SCRIPT_DIR/$PLIST_TEMPLATE" > "$LAUNCH_AGENTS_DIR/$PLIST_TEMPLATE"

echo "✓ Created plist file at: $LAUNCH_AGENTS_DIR/$PLIST_TEMPLATE"

# Unload if already loaded (ignore errors)
launchctl unload "$LAUNCH_AGENTS_DIR/$PLIST_TEMPLATE" 2>/dev/null || true

# Load the service
launchctl load "$LAUNCH_AGENTS_DIR/$PLIST_TEMPLATE"

echo ""
echo "✓ Service installed and started!"
echo ""
echo "Useful commands:"
echo "  Check status:  launchctl list | grep mlx-whisper"
echo "  Stop service:  launchctl stop $SERVICE_NAME"
echo "  Start service: launchctl start $SERVICE_NAME"
echo "  Restart:       launchctl kickstart -k gui/\$(id -u)/$SERVICE_NAME"
echo "  Uninstall:     ./uninstall_service.sh"
echo "  View logs:     tail -f $SCRIPT_DIR/server.log"
echo ""
echo "The service will automatically start on system reboot."