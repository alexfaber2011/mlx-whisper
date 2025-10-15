#!/usr/bin/env python3
"""Unit and integration tests for MLX Whisper Wyoming server."""
import asyncio
import json
import tempfile
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

# Import server components
from server import (
    ServerConfig,
    ServerMetrics,
    StructuredLogger,
    WhisperEventHandler,
    get_health_status,
    preload_model,
)


class TestServerConfig:
    """Test ServerConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 10300
        assert config.preload_model is True
        assert config.model_name == "mlx-community/whisper-large-v3-turbo"

    def test_from_yaml_missing_file(self, tmp_path):
        """Test loading config from non-existent file."""
        config_path = tmp_path / "missing.yaml"
        config = ServerConfig.from_yaml(config_path)
        # Should return defaults
        assert config.port == 10300

    def test_from_yaml_valid_file(self, tmp_path):
        """Test loading config from valid YAML file."""
        config_path = tmp_path / "test_config.yaml"
        config_data = """
server:
  host: "127.0.0.1"
  port: 9999
  preload_model: false

model:
  name: "mlx-community/whisper-tiny"
  language: "en"
  load_retry_count: 2
  load_retry_delay: 3

logging:
  level: "DEBUG"
  format: "json"
  include_request_ids: true

metrics:
  enabled: true

health_check:
  enabled: true
"""
        config_path.write_text(config_data)

        config = ServerConfig.from_yaml(config_path)
        assert config.host == "127.0.0.1"
        assert config.port == 9999
        assert config.preload_model is False
        assert config.model_name == "mlx-community/whisper-tiny"
        assert config.language == "en"
        assert config.log_level == "DEBUG"
        assert config.log_format == "json"


class TestServerMetrics:
    """Test ServerMetrics class."""

    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = ServerMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_transcriptions == 0
        assert metrics.failed_transcriptions == 0

    def test_to_dict(self):
        """Test metrics dictionary conversion."""
        metrics = ServerMetrics()
        metrics.total_requests = 10
        metrics.successful_transcriptions = 8
        metrics.failed_transcriptions = 2
        metrics.total_inference_time = 16.0
        metrics.total_audio_duration = 80.0
        metrics.active_connections = 3

        result = metrics.to_dict()
        assert result["total_requests"] == 10
        assert result["successful_transcriptions"] == 8
        assert result["failed_transcriptions"] == 2
        assert result["average_inference_time_seconds"] == 2.0
        assert result["active_connections"] == 3


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_text_logging(self, caplog):
        """Test text format logging."""
        import logging
        caplog.set_level(logging.INFO)
        logger = StructuredLogger("test", use_json=False)
        logger.info("Test message", key1="value1", key2="value2")

        assert "Test message" in caplog.text

    def test_json_logging(self, caplog):
        """Test JSON format logging."""
        import logging
        caplog.set_level(logging.INFO)
        logger = StructuredLogger("test", use_json=True)
        logger.info("Test message", key1="value1")

        # Check that JSON is logged
        assert "Test message" in caplog.text


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_status_no_model(self):
        """Test health status when model is not loaded."""
        import server
        server._MODEL_CACHE = None

        status = get_health_status()
        assert status["status"] == "degraded"
        assert status["model_loaded"] is False

    def test_health_status_with_model(self):
        """Test health status when model is loaded."""
        import server
        server._MODEL_CACHE = MagicMock()  # Mock model
        server._MODEL_LOADED_AT = 1000.0

        with patch('time.time', return_value=1010.0):
            status = get_health_status()
            assert status["status"] == "healthy"
            assert status["model_loaded"] is True
            assert status["uptime_seconds"] == 10.0


class TestWhisperEventHandler:
    """Test WhisperEventHandler class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ServerConfig(
            model_name="mlx-community/whisper-tiny",
            log_level="DEBUG",
            metrics_enabled=True,
        )

    @pytest.fixture
    def logger(self):
        """Create test logger."""
        return StructuredLogger("test", use_json=False)

    @pytest.fixture
    def mock_wyoming_info(self):
        """Create mock Wyoming info."""
        return MagicMock()

    def test_handler_initialization(self, config, logger, mock_wyoming_info):
        """Test handler initialization."""
        reader = MagicMock()
        writer = MagicMock()

        handler = WhisperEventHandler(
            reader, writer, mock_wyoming_info, config, logger
        )

        assert handler.config == config
        assert handler.logger == logger
        assert len(handler.request_id) == 8  # UUID prefix
        assert handler.audio_buffer == bytes()
        assert handler.transcribing is False


class TestAudioProcessing:
    """Test audio processing functionality."""

    def test_create_test_audio(self, tmp_path):
        """Test creation of test audio file."""
        audio_path = tmp_path / "test_audio.wav"

        # Create simple test audio
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)

        # Generate sine wave
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        # Verify file was created
        assert audio_path.exists()

        # Verify file properties
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == sample_rate
            assert wav_file.getsampwidth() == 2


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functions."""

    async def test_preload_model_disabled(self):
        """Test model preloading when disabled."""
        config = ServerConfig(preload_model=False)
        logger = StructuredLogger("test", use_json=False)

        result = await preload_model(config, logger)
        assert result is False

    async def test_preload_model_failure(self):
        """Test model preloading with failure."""
        config = ServerConfig(
            preload_model=True,
            load_retry_count=1,
            load_retry_delay=0
        )
        logger = StructuredLogger("test", use_json=False)

        with patch('mlx_whisper.load_models.load_model', side_effect=Exception("Load failed")):
            result = await preload_model(config, logger)
            assert result is False


class TestIntegration:
    """Integration tests."""

    @pytest.mark.integration
    def test_config_and_logging_integration(self, tmp_path):
        """Test configuration loading and logging integration."""
        config_path = tmp_path / "integration_config.yaml"
        config_data = """
server:
  host: "localhost"
  port: 11300

model:
  name: "mlx-community/whisper-tiny"

logging:
  level: "INFO"
  format: "text"
"""
        config_path.write_text(config_data)

        config = ServerConfig.from_yaml(config_path)
        logger = StructuredLogger("integration_test", use_json=(config.log_format == "json"))

        assert config.host == "localhost"
        assert config.port == 11300
        assert logger.use_json is False


def test_metrics_accumulation():
    """Test metrics accumulation over multiple requests."""
    metrics = ServerMetrics()

    # Simulate multiple requests
    for i in range(5):
        metrics.total_requests += 1
        metrics.successful_transcriptions += 1
        metrics.total_inference_time += 0.5
        metrics.total_audio_duration += 2.0

    result = metrics.to_dict()
    assert result["total_requests"] == 5
    assert result["successful_transcriptions"] == 5
    assert result["average_inference_time_seconds"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
