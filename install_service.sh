#!/bin/bash

# Install MLX Whisper STT Service for macOS

SERVICE_NAME="com.mlx.whisper.stt"
PLIST_FILE="${SERVICE_NAME}.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "Installing MLX Whisper STT Service..."

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy plist file to LaunchAgents
cp "$PLIST_FILE" "$LAUNCH_AGENTS_DIR/"

# Load the service
launchctl load "$LAUNCH_AGENTS_DIR/$PLIST_FILE"

# Start the service immediately
launchctl start "$SERVICE_NAME"

echo "Service installed and started!"
echo ""
echo "Useful commands:"
echo "  Check status:  launchctl list | grep $SERVICE_NAME"
echo "  Stop service:  launchctl stop $SERVICE_NAME"
echo "  Start service: launchctl start $SERVICE_NAME"
echo "  Uninstall:     launchctl unload $LAUNCH_AGENTS_DIR/$PLIST_FILE"
echo "  View logs:     tail -f server.log"
echo "  View errors:   tail -f server_error.log"