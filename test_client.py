#!/usr/bin/env python3
"""Simple test client for the MLX Whisper Wyoming server."""
import asyncio
import wave
from pathlib import Path

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient


async def test_transcription(audio_file: str, host: str = "localhost", port: int = 10300):
    """Test transcription of an audio file."""

    print(f"Connecting to {host}:{port}...")
    client = AsyncClient.from_uri(f"tcp://{host}:{port}")

    try:
        await client.connect()
        print("Connected!")

        # Read audio file info first
        print(f"Reading audio from {audio_file}...")
        with wave.open(audio_file, "rb") as wav_file:
            rate = wav_file.getframerate()
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            print(f"Audio format: {rate}Hz, {width} bytes/sample, {channels} channel(s)")

        # Send transcribe request
        print("Sending transcribe request...")
        await client.write_event(Transcribe().event())

        # Send audio start
        await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())

        # Read and send audio file in chunks
        print(f"Sending audio chunks...")
        with wave.open(audio_file, "rb") as wav_file:

            chunk_size = 1024
            while True:
                frames = wav_file.readframes(chunk_size)
                if not frames:
                    break

                chunk = AudioChunk(
                    rate=rate,
                    width=width,
                    channels=channels,
                    audio=frames,
                )
                await client.write_event(chunk.event())

        # Send audio stop
        print("Audio sent, waiting for transcription...")
        await client.write_event(AudioStop().event())

        # Wait for transcript
        while True:
            event = await client.read_event()
            if event is None:
                break

            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                print(f"\nTranscription result: '{transcript.text}'")
                break

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error during test: {e}")
        raise
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_client.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    asyncio.run(test_transcription(audio_file))
