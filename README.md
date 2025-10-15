# MLX Whisper Wyoming Protocol Server

A production-ready Wyoming protocol-compatible speech-to-text server using MLX Whisper, optimized for Apple Silicon Macs.

## Overview

This server implements the Wyoming protocol for automatic speech recognition (ASR) using MLX Whisper, providing fast, on-device transcription that leverages Apple's Metal framework for acceleration on M-series chips.

### Features

- **Wyoming Protocol Compatible** - Works seamlessly with Home Assistant and other Wyoming clients
- **Apple Silicon Optimized** - Leverages MLX framework for fast inference on M1/M2/M3/M4 chips
- **YAML Configuration** - Flexible configuration via `config.yaml` with CLI overrides
- **Model Preloading** - Optional model preloading at startup for instant first transcription
- **Structured Logging** - JSON or text logging with request IDs for traceability
- **Performance Metrics** - Built-in metrics tracking (requests, latency, throughput)
- **Health Check** - Monitor server and model status
- **Error Handling** - Automatic retry logic with graceful degradation
- **Production Ready** - Type hints, comprehensive tests, and error handling

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone or download this repository**

3. **Install dependencies**:
```bash
uv sync
```

4. **Configure the server** (optional):
Edit `config.yaml` to customize settings like port, model, logging format, etc.

## Running the Server

### Quick Start (Foreground)

Run the server directly:
```bash
uv run python server.py
```

Or with custom options (CLI arguments override config.yaml):
```bash
uv run python server.py --host 0.0.0.0 --port 10300 --model mlx-community/whisper-large-v3-turbo
```

### Command Line Options

- `--config`: Path to configuration file (default: `config.yaml`)
- `--host`: Host to bind to (overrides config)
- `--port`: Port to bind to (overrides config)
- `--model`: Whisper model to use (overrides config)
- `--language`: Default language code (overrides config, e.g., `en`, `es`, `fr`)
- `--debug`: Enable debug logging (sets log level to DEBUG)

### Configuration File

The server reads settings from `config.yaml` by default. Key settings:

```yaml
server:
  host: "0.0.0.0"
  port: 10300
  preload_model: true  # Load model at startup

model:
  name: "mlx-community/whisper-large-v3-turbo"
  language: null  # null = auto-detect
  load_retry_count: 3  # Retry on failure
  load_retry_delay: 5  # Seconds between retries

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "text"  # "text" or "json"
  include_request_ids: true

metrics:
  enabled: true
  track_inference_time: true
  track_audio_duration: true
```

See `config.yaml` for all available options.

## Running in Background

### Using launchd (Recommended)

Create a LaunchAgent to run the server automatically in the background and restart on system reboot.

#### Option 1: Automated Installation (Easiest)

Use the provided installation script which automatically detects paths and configures the service:

```bash
# Make scripts executable
chmod +x install_service.sh uninstall_service.sh

# Install and start the service
./install_service.sh
```

The script will:
- Auto-detect your `uv` installation path
- Auto-detect your `ffmpeg` location (for audio processing)
- Configure the service with the correct working directory
- Install and start the service automatically

To uninstall:
```bash
./uninstall_service.sh
```

#### Option 2: Manual Installation

If you prefer manual setup, follow these steps:

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
4. Enter your Mac's IP address and port (10300)

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest test_server.py -v

# Run with coverage
uv run pytest test_server.py -v --cov=server

# Run specific test
uv run pytest test_server.py::TestServerConfig -v
```

## Performance Monitoring

The server tracks performance metrics automatically when enabled in config.yaml:

- Total requests processed
- Successful vs failed transcriptions
- Average inference time
- Real-time factor (inference time / audio duration)
- Active connections

Metrics are logged with each transcription. For JSON logging, enable it in config:

```yaml
logging:
  format: "json"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Wyoming Client                     │
│              (Home Assistant, etc.)                  │
└──────────────────┬──────────────────────────────────┘
                   │ TCP Connection
                   │ Wyoming Protocol
┌──────────────────▼──────────────────────────────────┐
│              AsyncServer (server.py)                 │
│  ┌────────────────────────────────────────────────┐ │
│  │   WhisperEventHandler (per connection)         │ │
│  │   - Handles Wyoming protocol events            │ │
│  │   - Manages audio buffering                    │ │
│  │   - Request ID tracking                        │ │
│  │   - Metrics collection                         │ │
│  └────────────────┬───────────────────────────────┘ │
└───────────────────┼───────────────────────────────── ┘
                    │
                    │ Loads on startup (if preload=true)
                    │ or on first request
┌───────────────────▼───────────────────────────────┐
│            MLX Whisper Model                      │
│       (Cached globally, shared across             │
│             all connections)                      │
└───────────────────────────────────────────────────┘
```

### Key Components

- **ServerConfig**: YAML configuration loader with CLI overrides
- **StructuredLogger**: Supports both text and JSON logging with request IDs
- **ServerMetrics**: Thread-safe metrics collection
- **WhisperEventHandler**: Per-connection Wyoming protocol handler
- **Model Cache**: Global model instance shared across all connections

## Troubleshooting

### Server won't start
- Check if port 10300 is already in use: `lsof -i :10300`
- Verify Python dependencies are installed: `uv sync`
- Check logs for errors
- Verify config.yaml syntax is valid

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
- Check retry settings in config.yaml if models fail to load

### Permission denied (launchd)
- Ensure the plist file has correct permissions: `chmod 644 ~/Library/LaunchAgents/com.mlx-whisper.wyoming.plist`
- Check paths are absolute and correct
- Verify PATH includes `/opt/homebrew/bin` for ffmpeg access

### Service not auto-starting on reboot
- Verify the plist is loaded: `launchctl list | grep mlx-whisper`
- Check system logs: `log show --predicate 'process == "launchd"' --last 1h`
- Ensure `RunAtLoad` is set to `true` in the plist file

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

## What's New in v2.0

This version includes significant enhancements for production use:

### Configuration & Flexibility
- YAML-based configuration with CLI overrides
- Model preloading option for faster first transcription
- Configurable retry logic for model loading

### Observability
- Structured logging with JSON output support
- Request ID tracking for distributed tracing
- Built-in performance metrics (latency, throughput, real-time factor)
- Health check function for monitoring

### Reliability
- Automatic retry with exponential backoff
- Audio buffer size limits
- Graceful error handling and recovery
- Global model caching to reduce memory usage

### Developer Experience
- Comprehensive type hints throughout
- Unit and integration test suite
- Better error messages and logging

## API Reference

For detailed API documentation, see the inline docstrings in `server.py`:

- `ServerConfig`: Configuration management
- `ServerMetrics`: Performance tracking
- `StructuredLogger`: Flexible logging
- `WhisperEventHandler`: Wyoming protocol handler
- `preload_model()`: Model preloading function
- `get_health_status()`: Health check endpoint

## Contributing

Contributions are welcome! Please ensure:
- All tests pass: `pytest test_server.py -v`
- Code follows existing style
- Type hints are included
- Documentation is updated

## License

MIT License - See LICENSE file for details
