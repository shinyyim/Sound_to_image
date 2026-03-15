"""
Module 01 — Ambisonic Audio Loader
Loads FOA (First-Order Ambisonics) audio files with 4 channels: W, X, Y, Z.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def load_audio(filepath: str, target_sr: int = 24000) -> dict:
    """Load an audio file and return audio data with metadata.

    Args:
        filepath: Path to audio file (WAV, FLAC, OGG)
        target_sr: Target sample rate (default 24000 for TAU/STARSS compatibility)

    Returns:
        dict with keys: audio, sr, channels, duration, filepath, is_foa
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    audio, sr = sf.read(str(filepath), always_2d=True)

    # Resample if needed
    if sr != target_sr:
        import librosa
        audio_resampled = []
        for ch in range(audio.shape[1]):
            resampled = librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr)
            audio_resampled.append(resampled)
        audio = np.stack(audio_resampled, axis=1)
        sr = target_sr

    n_channels = audio.shape[1]
    duration = audio.shape[0] / sr
    is_foa = n_channels == 4

    return {
        "audio": audio,           # shape: (samples, channels)
        "sr": sr,
        "channels": n_channels,
        "duration": duration,
        "filepath": str(filepath),
        "is_foa": is_foa,
        "channel_labels": ["W", "X", "Y", "Z"][:n_channels],
    }


def validate_foa(audio_data: dict) -> bool:
    """Validate that audio is proper FOA format (4 channels)."""
    if not audio_data["is_foa"]:
        print(f"WARNING: Expected 4-channel FOA, got {audio_data['channels']} channels")
        return False
    return True


def get_channel(audio_data: dict, channel: str) -> np.ndarray:
    """Extract a single channel by label (W, X, Y, or Z)."""
    labels = audio_data["channel_labels"]
    if channel not in labels:
        raise ValueError(f"Channel '{channel}' not found. Available: {labels}")
    idx = labels.index(channel)
    return audio_data["audio"][:, idx]


def run(filepath: str, target_sr: int = 24000) -> dict:
    """Entry point for the capture module."""
    audio_data = load_audio(filepath, target_sr)
    validate_foa(audio_data)
    print(f"Loaded: {filepath}")
    print(f"  Channels: {audio_data['channels']} ({', '.join(audio_data['channel_labels'])})")
    print(f"  Duration: {audio_data['duration']:.2f}s | Sample rate: {audio_data['sr']} Hz")
    print(f"  FOA: {audio_data['is_foa']}")
    return audio_data
