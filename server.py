#!/usr/bin/env python3
"""MLX Whisper Wyoming Protocol Server with enhanced features."""
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Ensure ffmpeg is in PATH (needed for mlx_whisper)
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"

import mlx_whisper
import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


@dataclass
class ServerMetrics:
    """Track server performance metrics."""
    total_requests: int = 0
    successful_transcriptions: int = 0
    failed_transcriptions: int = 0
    total_inference_time: float = 0.0
    total_audio_duration: float = 0.0
    active_connections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        avg_inference_time = (
            self.total_inference_time / self.successful_transcriptions
            if self.successful_transcriptions > 0
            else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "average_inference_time_seconds": round(avg_inference_time, 3),
            "total_audio_duration_seconds": round(self.total_audio_duration, 2),
            "active_connections": self.active_connections,
        }


class StructuredLogger:
    """Logger that supports both JSON and text output."""

    def __init__(self, name: str, use_json: bool = False):
        self.logger = logging.getLogger(name)
        self.use_json = use_json

    def _log(self, level: int, message: str, **kwargs):
        """Log with optional JSON formatting."""
        if self.use_json:
            log_data = {
                "timestamp": time.time(),
                "message": message,
                **kwargs
            }
            self.logger.log(level, json.dumps(log_data))
        else:
            extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
            full_message = f"{message} {extra_info}" if extra_info else message
            self.logger.log(level, full_message)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 10300
    preload_model: bool = True
    model_name: str = "mlx-community/whisper-large-v3-turbo"
    language: Optional[str] = None
    load_retry_count: int = 3
    load_retry_delay: int = 5
    max_buffer_size_mb: int = 100
    log_level: str = "INFO"
    log_format: str = "text"
    include_request_ids: bool = True
    metrics_enabled: bool = True
    health_check_enabled: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ServerConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            _LOGGER.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            host=data.get("server", {}).get("host", "0.0.0.0"),
            port=data.get("server", {}).get("port", 10300),
            preload_model=data.get("server", {}).get("preload_model", True),
            model_name=data.get("model", {}).get("name", "mlx-community/whisper-large-v3-turbo"),
            language=data.get("model", {}).get("language"),
            load_retry_count=data.get("model", {}).get("load_retry_count", 3),
            load_retry_delay=data.get("model", {}).get("load_retry_delay", 5),
            max_buffer_size_mb=data.get("audio", {}).get("max_buffer_size_mb", 100),
            log_level=data.get("logging", {}).get("level", "INFO"),
            log_format=data.get("logging", {}).get("format", "text"),
            include_request_ids=data.get("logging", {}).get("include_request_ids", True),
            metrics_enabled=data.get("metrics", {}).get("enabled", True),
            health_check_enabled=data.get("health_check", {}).get("enabled", True),
        )


# Global metrics and model cache
_METRICS = ServerMetrics()
_MODEL_CACHE: Optional[Any] = None
_MODEL_LOADED_AT: Optional[float] = None

class WhisperEventHandler(AsyncEventHandler):
    """Wyoming event handler for MLX Whisper ASR with enhanced features."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        config: ServerConfig,
        logger: StructuredLogger,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)

        self.wyoming_info = wyoming_info
        self.config = config
        self.logger = logger
        self.request_id = str(uuid.uuid4())[:8]
        self.converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self.audio_buffer = bytes()
        self.transcribing = False
        self.transcribe_start_time: Optional[float] = None

        # Use global model cache
        global _METRICS
        _METRICS.active_connections += 1

        self.logger.info(
            "Handler initialized",
            request_id=self.request_id,
            model=config.model_name
        )

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        global _METRICS
        _METRICS.active_connections -= 1

    async def load_model(self) -> Any:
        """Load the Whisper model with retry logic."""
        global _MODEL_CACHE, _MODEL_LOADED_AT

        # Use cached model if available
        if _MODEL_CACHE is not None:
            self.logger.debug(
                "Using cached model",
                request_id=self.request_id
            )
            return _MODEL_CACHE

        # Load model with retries
        for attempt in range(self.config.load_retry_count):
            try:
                self.logger.info(
                    "Loading Whisper model",
                    request_id=self.request_id,
                    model=self.config.model_name,
                    attempt=attempt + 1
                )

                model = await asyncio.to_thread(
                    mlx_whisper.load_models.load_model,
                    self.config.model_name
                )

                _MODEL_CACHE = model
                _MODEL_LOADED_AT = time.time()

                self.logger.info(
                    "Model loaded successfully",
                    request_id=self.request_id
                )
                return model

            except Exception as e:
                self.logger.error(
                    "Failed to load model",
                    request_id=self.request_id,
                    error=str(e),
                    attempt=attempt + 1
                )

                if attempt < self.config.load_retry_count - 1:
                    await asyncio.sleep(self.config.load_retry_delay)
                else:
                    raise

        raise RuntimeError("Failed to load model after all retries")

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events with enhanced logging."""
        self.logger.debug(
            "Received event",
            request_id=self.request_id,
            event_type=event.type
        )

        try:
            if Info.is_type(event.type) or event.type == "describe":
                self.logger.debug(
                    "Sending Wyoming info",
                    request_id=self.request_id
                )
                await self.write_event(self.wyoming_info.event())
                return True

            if AudioChunk.is_type(event.type):
                if not self.transcribing:
                    return True

                chunk = AudioChunk.from_event(event)
                chunk = self.converter.convert(chunk)

                # Check buffer size limit
                new_size = len(self.audio_buffer) + len(chunk.audio)
                max_size = self.config.max_buffer_size_mb * 1024 * 1024

                if new_size > max_size:
                    self.logger.warning(
                        "Audio buffer size exceeded",
                        request_id=self.request_id,
                        size_mb=round(new_size / 1024 / 1024, 2),
                        max_mb=self.config.max_buffer_size_mb
                    )
                    # Truncate or reject
                    return True

                self.audio_buffer += chunk.audio
                return True

            if AudioStop.is_type(event.type):
                self.logger.debug(
                    "Received AudioStop",
                    request_id=self.request_id
                )
                self.transcribing = False

                if self.audio_buffer:
                    await self.transcribe_audio()

                self.audio_buffer = bytes()
                return True

            if Transcribe.is_type(event.type):
                global _METRICS
                _METRICS.total_requests += 1

                self.logger.info(
                    "Received Transcribe request",
                    request_id=self.request_id
                )
                transcribe = Transcribe.from_event(event)

                language = transcribe.language or self.config.language
                if language:
                    self.logger.debug(
                        "Language set",
                        request_id=self.request_id,
                        language=language
                    )

                self.transcribing = True
                self.audio_buffer = bytes()
                self.transcribe_start_time = time.time()

                # Ensure model is loaded
                await self.load_model()
                return True

        except Exception as e:
            self.logger.error(
                "Error handling event",
                request_id=self.request_id,
                error=str(e),
                event_type=event.type
            )
            return False

        return True

    async def transcribe_audio(self):
        """Transcribe the buffered audio with metrics tracking."""
        global _METRICS, _MODEL_CACHE

        if _MODEL_CACHE is None:
            self.logger.error(
                "Model not loaded",
                request_id=self.request_id
            )
            _METRICS.failed_transcriptions += 1
            transcript = Transcript(text="")
            await self.write_event(transcript.event())
            return

        tmp_wav_path = None
        try:
            # Calculate audio duration for metrics
            audio_duration = len(self.audio_buffer) / (16000 * 2)  # samples / (rate * bytes_per_sample)

            # Write audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                with wave.open(tmp_wav.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(self.audio_buffer)
                tmp_wav_path = tmp_wav.name

            self.logger.debug(
                "Transcribing audio",
                request_id=self.request_id,
                file=tmp_wav_path,
                duration_seconds=round(audio_duration, 2)
            )

            # Transcribe with timing
            inference_start = time.time()

            options = {}
            if self.config.language:
                options['language'] = self.config.language

            result = await asyncio.to_thread(
                mlx_whisper.transcribe,
                tmp_wav_path,
                path_or_hf_repo=self.config.model_name,
                verbose=False,
                **options
            )

            inference_time = time.time() - inference_start

            # Extract text
            text_list = result.get("text", [])
            text = "".join(text_list).strip() if isinstance(text_list, list) else str(text_list).strip()

            self.logger.info(
                "Transcription complete",
                request_id=self.request_id,
                text=text[:100] + "..." if len(text) > 100 else text,
                inference_time_seconds=round(inference_time, 3),
                audio_duration_seconds=round(audio_duration, 2),
                realtime_factor=round(inference_time / audio_duration, 2) if audio_duration > 0 else 0
            )

            # Update metrics
            if self.config.metrics_enabled:
                _METRICS.successful_transcriptions += 1
                _METRICS.total_inference_time += inference_time
                _METRICS.total_audio_duration += audio_duration

            # Send response
            transcript = Transcript(text=text)
            await self.write_event(transcript.event())

        except Exception as e:
            self.logger.error(
                "Transcription error",
                request_id=self.request_id,
                error=str(e)
            )
            _METRICS.failed_transcriptions += 1

            # Send empty transcript on error
            transcript = Transcript(text="")
            await self.write_event(transcript.event())

        finally:
            # Clean up temp file
            if tmp_wav_path:
                Path(tmp_wav_path).unlink(missing_ok=True)


async def preload_model(config: ServerConfig, logger: StructuredLogger) -> bool:
    """Preload the Whisper model at startup."""
    global _MODEL_CACHE, _MODEL_LOADED_AT

    if not config.preload_model:
        logger.info("Model preloading disabled")
        return False

    logger.info("Preloading model", model=config.model_name)

    for attempt in range(config.load_retry_count):
        try:
            model = await asyncio.to_thread(
                mlx_whisper.load_models.load_model,
                config.model_name
            )
            _MODEL_CACHE = model
            _MODEL_LOADED_AT = time.time()
            logger.info("Model preloaded successfully")
            return True

        except Exception as e:
            logger.error(
                "Failed to preload model",
                error=str(e),
                attempt=attempt + 1
            )
            if attempt < config.load_retry_count - 1:
                await asyncio.sleep(config.load_retry_delay)

    logger.warning("Could not preload model, will load on first request")
    return False


def get_health_status() -> Dict[str, Any]:
    """Get server health status."""
    global _MODEL_CACHE, _MODEL_LOADED_AT, _METRICS

    status = {
        "status": "healthy",
        "model_loaded": _MODEL_CACHE is not None,
        "uptime_seconds": round(time.time() - _MODEL_LOADED_AT, 2) if _MODEL_LOADED_AT else None,
        "metrics": _METRICS.to_dict()
    }

    if _MODEL_CACHE is None:
        status["status"] = "degraded"

    return status


async def main():
    """Main entry point with enhanced configuration and health check."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MLX Whisper Wyoming Protocol Server"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--host",
        help="Host to bind to (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (overrides config)",
    )
    parser.add_argument(
        "--model",
        help="Whisper model to use (overrides config)",
    )
    parser.add_argument(
        "--language",
        help="Default language code (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = ServerConfig.from_yaml(args.config)

    # Apply command-line overrides
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.model:
        config.model_name = args.model
    if args.language:
        config.language = args.language
    if args.debug:
        config.log_level = "DEBUG"

    # Setup logging
    log_level = getattr(logging, config.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = StructuredLogger(__name__, use_json=(config.log_format == "json"))
    logger.info(
        "Starting MLX Whisper Wyoming Server",
        host=config.host,
        port=config.port,
        model=config.model_name,
        config_file=str(args.config)
    )

    # Preload model if configured
    if config.preload_model:
        await preload_model(config, logger)

    # Create Wyoming info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="mlx-whisper",
                description="MLX Whisper Speech-to-Text",
                attribution=Attribution(
                    name="MLX Whisper",
                    url="https://github.com/ml-explore/mlx",
                ),
                installed=True,
                version="2.0.0",
                models=[
                    AsrModel(
                        name=config.model_name.split("/")[-1],
                        description=f"Whisper {config.model_name.split('/')[-1]} model",
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        languages=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"],
                        version="1.0.0",
                    )
                ],
            )
        ],
    )

    logger.info("Server ready", status="listening")

    # Create server
    server = AsyncServer.from_uri(f"tcp://{config.host}:{config.port}")

    def make_handler(reader, writer):
        handler_logger = StructuredLogger(
            f"{__name__}.handler",
            use_json=(config.log_format == "json")
        )
        return WhisperEventHandler(
            reader,
            writer,
            wyoming_info,
            config,
            handler_logger,
        )

    try:
        await server.run(make_handler)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
    except Exception as e:
        logger.error("Server error", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())