#!/usr/bin/env python3
import asyncio
import io
import logging
import tempfile
import wave
from pathlib import Path
from typing import Optional

import mlx_whisper
import numpy as np
from pydub import AudioSegment
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)

class WhisperEventHandler(AsyncEventHandler):
    """Wyoming event handler for MLX Whisper ASR."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        model_name: str = "mlx-community/whisper-large-v3-turbo",
        language: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)
        
        self.wyoming_info = wyoming_info
        self.model_name = model_name
        self.language = language
        self.model = None
        self.converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self.audio_buffer = bytes()
        self.transcribing = False
        
        _LOGGER.info(f"Initialized handler with model: {model_name}")

    async def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            _LOGGER.info(f"Loading Whisper model: {self.model_name}")
            try:
                self.model = await asyncio.to_thread(
                    mlx_whisper.load_models.load_model, self.model_name
                )
                _LOGGER.info("Model loaded successfully")
            except Exception as e:
                _LOGGER.error(f"Failed to load model: {e}")
                raise

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""
        
        _LOGGER.debug(f"Received event: {event.type}")
        
        if Info.is_type(event.type) or event.type == "describe":
            _LOGGER.debug("Sending Wyoming info")
            await self.write_event(self.wyoming_info.event())
            return True
            
        if AudioChunk.is_type(event.type):
            if not self.transcribing:
                return True
                
            chunk = AudioChunk.from_event(event)
            chunk = self.converter.convert(chunk)
            self.audio_buffer += chunk.audio
            return True
            
        if AudioStop.is_type(event.type):
            _LOGGER.debug("Received AudioStop")
            self.transcribing = False
            
            if self.audio_buffer:
                await self.transcribe_audio()
                
            self.audio_buffer = bytes()
            return True
            
        if Transcribe.is_type(event.type):
            _LOGGER.debug("Received Transcribe request")
            transcribe = Transcribe.from_event(event)
            
            if transcribe.language:
                self.language = transcribe.language
                _LOGGER.debug(f"Language set to: {self.language}")
                
            self.transcribing = True
            self.audio_buffer = bytes()
            
            await self.load_model()
            return True
            
        return True

    async def transcribe_audio(self):
        """Transcribe the buffered audio."""
        if not self.model:
            _LOGGER.error("Model not loaded")
            return
            
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                with wave.open(tmp_wav.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(self.audio_buffer)
                
                tmp_wav_path = tmp_wav.name
            
            _LOGGER.debug(f"Transcribing audio file: {tmp_wav_path}")
            
            options = {}
            if self.language:
                options['language'] = self.language
            
            result = await asyncio.to_thread(
                mlx_whisper.transcribe,
                tmp_wav_path,
                path_or_hf_repo=self.model_name,
                verbose=False,
                **options
            )
            
            text = result.get("text", "").strip()
            _LOGGER.info(f"Transcription: {text}")
            
            transcript = Transcript(text=text)
            await self.write_event(transcript.event())
            
            Path(tmp_wav_path).unlink(missing_ok=True)
            
        except Exception as e:
            _LOGGER.error(f"Transcription error: {e}")
            transcript = Transcript(text="")
            await self.write_event(transcript.event())


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10300,
        help="Port to bind to",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model to use",
    )
    parser.add_argument(
        "--language",
        help="Default language code (e.g., en, es, fr)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
                version="1.0.0",
                models=[
                    AsrModel(
                        name=args.model.split("/")[-1],
                        description=f"Whisper {args.model.split('/')[-1]} model",
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
    
    _LOGGER.info(f"Starting Wyoming server on {args.host}:{args.port}")
    
    server = AsyncServer.from_uri(
        f"tcp://{args.host}:{args.port}"
    )
    
    def make_handler(reader, writer):
        return WhisperEventHandler(
            reader,
            writer,
            wyoming_info,
            model_name=args.model,
            language=args.language,
        )
    
    try:
        await server.run(make_handler)
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down")


if __name__ == "__main__":
    from functools import partial
    
    asyncio.run(main())