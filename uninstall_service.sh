#!/bin/bash

# Uninstall MLX Whisper STT Service

SERVICE_NAME="com.mlx.whisper.stt"
PLIST_FILE="${SERVICE_NAME}.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "Uninstalling MLX Whisper STT Service..."

# Stop the service
launchctl stop "$SERVICE_NAME"

# Unload the service
launchctl unload "$LAUNCH_AGENTS_DIR/$PLIST_FILE"

# Remove the plist file
rm -f "$LAUNCH_AGENTS_DIR/$PLIST_FILE"

echo "Service uninstalled successfully!"