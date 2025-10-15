# MLX Whisper Wyoming Protocol Server

A Wyoming protocol-compatible speech-to-text server using MLX Whisper, optimized for Apple Silicon Macs.

## Overview

This server implements the Wyoming protocol for automatic speech recognition (ASR) using MLX Whisper, providing fast, on-device transcription that leverages Apple's Metal framework for acceleration on M-series chips.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies**:
```bash
uv sync
```

Or using pip:
```bash
pip install -r pyproject.toml
```

## Running the Server

### Quick Start (Foreground)

Run the server directly:
```bash
uv run python server.py
```

Or with custom options:
```bash
uv run python server.py --host 0.0.0.0 --port 10300 --model mlx-community/whisper-large-v3-turbo
```

### Command Line Options

- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to bind to (default: `10300`)
- `--model`: Whisper model to use (default: `mlx-community/whisper-large-v3-turbo`)
- `--language`: Default language code (e.g., `en`, `es`, `fr`)
- `--debug`: Enable debug logging

## Running in Background on Mac Studio

### Option 1: Using launchd (Recommended for macOS)

Create a LaunchAgent to run the server automatically in the background and restart on system reboot.

1. **Create a plist file** at `~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx-whisper.wyoming</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/alex/.local/bin/uv</string>
        <string>run</string>
        <string>python</string>
        <string>server.py</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>10300</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/alex/python/mlx-project</string>

    <key>StandardOutPath</key>
    <string>/Users/alex/python/mlx-project/server.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/alex/python/mlx-project/server.log</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/Users/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

**Note**: Adjust the paths if your `uv` installation or project directory is different. Find your uv path with `which uv`. The PATH must include `/opt/homebrew/bin` for ffmpeg access.

2. **Load the service**:
```bash
launchctl load ~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist
```

3. **Start the service**:
```bash
launchctl start com.mlx-whisper.wyoming
```

4. **Check if it's running**:
```bash
launchctl list | grep mlx-whisper
```

5. **View logs**:
```bash
tail -f ~/python/mlx-project/server.log
```

6. **Stop the service**:
```bash
launchctl stop com.mlx-whisper.wyoming
```

7. **Unload the service** (to prevent auto-start):
```bash
launchctl unload ~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist
```

### Option 2: Using nohup (Simple Background Process)

For a simple background process without auto-restart:

1. **Start in background**:
```bash
nohup uv run python server.py > server.log 2>&1 &
```

2. **Save the process ID** (PID):
```bash
echo $! > server.pid
```

3. **Check if running**:
```bash
ps aux | grep server.py
```

4. **View logs**:
```bash
tail -f server.log
```

5. **Stop the server**:
```bash
kill $(cat server.pid)
```

Or find and kill manually:
```bash
pkill -f "python server.py"
```

### Option 3: Using screen or tmux (Terminal Multiplexer)

For persistent terminal sessions:

**Using screen**:
```bash
screen -S mlx-whisper
uv run python server.py
# Press Ctrl+A then D to detach
```

Reattach later:
```bash
screen -r mlx-whisper
```

**Using tmux**:
```bash
tmux new -s mlx-whisper
uv run python server.py
# Press Ctrl+B then D to detach
```

Reattach later:
```bash
tmux attach -t mlx-whisper
```

## Testing the Server

### Quick Port Check

Verify the server is listening:

```bash
# Check if port is open
nc -zv localhost 10300

# Or use lsof
lsof -i :10300
```

### Full Transcription Test

Test actual transcription functionality using the included test client:

```bash
# Test with any WAV file
uv run python test_client.py <path-to-audio.wav>

# Example with test.wav
uv run python test_client.py test.wav
```

**Expected output:**
```
Connecting to localhost:10300...
Connected!
Reading audio from test.wav...
Audio format: 16000Hz, 2 bytes/sample, 1 channel(s)
Sending transcribe request...
Sending audio chunks...
Audio sent, waiting for transcription...

Transcription result: 'Your transcribed text here'
Test completed successfully!
```

**Audio requirements:**
- Format: WAV file
- The server automatically converts audio to 16kHz, 16-bit, mono format
- Any sample rate and channel configuration is supported

## Integration with Home Assistant

This server is compatible with Home Assistant's Wyoming protocol integration:

1. In Home Assistant, go to Settings � Devices & Services
2. Click "Add Integration"
3. Search for "Wyoming Protocol"
4. Enter your Mac Studio's IP address and port (10300)

## Troubleshooting

### Server won't start
- Check if port 10300 is already in use: `lsof -i :10300`
- Verify Python dependencies are installed: `uv sync`
- Check logs for errors

### Transcription fails with "No such file or directory: 'ffmpeg'"
This happens when ffmpeg is not in the PATH used by launchd. To fix:

1. Find your ffmpeg location: `which ffmpeg` (typically `/opt/homebrew/bin/ffmpeg`)
2. Edit `~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist`
3. Update the PATH in the `EnvironmentVariables` section to include `/opt/homebrew/bin`:
   ```xml
   <key>PATH</key>
   <string>/opt/homebrew/bin:/Users/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
   ```
4. Reload the service:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist
   launchctl load ~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist
   ```

### Model loading errors
- Ensure you have enough disk space for model downloads
- First run will download the model (may take a few minutes)
- Models are cached in `~/.cache/huggingface/`

### Permission denied (launchd)
- Ensure the plist file has correct permissions: `chmod 644 ~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist`
- Check paths are absolute and correct

### Service not auto-starting on reboot
- Verify the plist is loaded: `launchctl list | grep mlx-whisper`
- Check system logs: `log show --predicate 'process == "launchd"' --last 1h`

## Performance Notes

- First transcription may be slower due to model loading
- Optimized for Apple Silicon using MLX framework
- Large models provide better accuracy but use more memory
- Default model (whisper-large-v3-turbo) offers good balance of speed and accuracy

## Available Models

- `mlx-community/whisper-tiny` - Fastest, least accurate
- `mlx-community/whisper-base` - Fast, good for simple use
- `mlx-community/whisper-small` - Balanced
- `mlx-community/whisper-medium` - More accurate
- `mlx-community/whisper-large-v3-turbo` - Very accurate, fast (default)
- `mlx-community/whisper-large-v3` - Most accurate, slower

Change the model by modifying the `--model` argument in your startup command or plist file.

## License

[Add your license here]
