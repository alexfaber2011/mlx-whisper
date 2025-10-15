#!/bin/bash

# Uninstall MLX Whisper Wyoming Protocol Service

SERVICE_NAME="com.mlx-whisper.wyoming"
PLIST_FILE="${SERVICE_NAME}.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "Uninstalling MLX Whisper Wyoming Protocol Service..."

# Stop the service (ignore errors if not running)
launchctl stop "$SERVICE_NAME" 2>/dev/null || true

# Unload the service (ignore errors if not loaded)
launchctl unload "$LAUNCH_AGENTS_DIR/$PLIST_FILE" 2>/dev/null || true

# Remove the plist file
if [ -f "$LAUNCH_AGENTS_DIR/$PLIST_FILE" ]; then
    rm -f "$LAUNCH_AGENTS_DIR/$PLIST_FILE"
    echo "✓ Removed $LAUNCH_AGENTS_DIR/$PLIST_FILE"
else
    echo "⚠ Plist file not found: $LAUNCH_AGENTS_DIR/$PLIST_FILE"
fi

echo ""
echo "✓ Service uninstalled successfully!"
echo ""
echo "Note: Log files in the project directory were not removed."
echo "To remove logs: rm -f server.log"