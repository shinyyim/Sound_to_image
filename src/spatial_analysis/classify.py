"""
Module 02b — YAMNet Sound Event Classification
Classifies sound events in audio using Google's YAMNet model via TensorFlow Hub.

YAMNet expects 16 kHz mono audio and produces per-frame (0.48 s hop) class scores
over 521 AudioSet classes.

This module:
  - Loads YAMNet lazily from tensorflow-hub
  - Accepts the audio_data dict produced by src.capture.loader
  - Returns top-N detected sound classes with confidence, grouped by time segment
  - Falls back gracefully when tensorflow is not installed
"""

import warnings
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# YAMNet constants
# ---------------------------------------------------------------------------
_YAMNET_SR = 16000          # YAMNet's required sample rate
_YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
_YAMNET_FRAME_HOP_SEC = 0.48  # Each YAMNet frame is ~0.48 s apart

# ---------------------------------------------------------------------------
# Lazy loader — keeps model in module-level cache so we load once
# ---------------------------------------------------------------------------
_yamnet_model = None
_yamnet_class_names = None


def _load_yamnet():
    """Load YAMNet model and class map. Raises ImportError if TF unavailable."""
    global _yamnet_model, _yamnet_class_names

    if _yamnet_model is not None:
        return _yamnet_model, _yamnet_class_names

    import tensorflow as tf
    import tensorflow_hub as hub
    import csv, io, urllib.request

    print("Loading YAMNet model from TensorFlow Hub...")
    _yamnet_model = hub.load(_YAMNET_HUB_URL)

    # The model's class_map.csv is bundled inside the SavedModel asset dir.
    # We can also fetch it directly — more robust across hub versions.
    class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        reader = csv.DictReader(f)
        _yamnet_class_names = [row["display_name"] for row in reader]

    print(f"YAMNet loaded — {len(_yamnet_class_names)} classes available.")
    return _yamnet_model, _yamnet_class_names


# ---------------------------------------------------------------------------
# Audio pre-processing
# ---------------------------------------------------------------------------

def _prepare_audio(audio_data: dict) -> np.ndarray:
    """Convert audio_data to 16 kHz mono float32 waveform expected by YAMNet.

    Args:
        audio_data: dict from src.capture.loader (keys: audio, sr, ...)

    Returns:
        1-D numpy float32 array at 16 kHz
    """
    audio = audio_data["audio"]  # (samples, channels)
    sr = audio_data["sr"]

    # Mix to mono — simple average across channels
    if audio.ndim == 2 and audio.shape[1] > 1:
        mono = np.mean(audio, axis=1)
    elif audio.ndim == 2:
        mono = audio[:, 0]
    else:
        mono = audio

    # Resample to 16 kHz if needed
    if sr != _YAMNET_SR:
        try:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=_YAMNET_SR)
        except ImportError:
            # Manual linear interpolation fallback (lower quality but no deps)
            ratio = _YAMNET_SR / sr
            n_out = int(len(mono) * ratio)
            indices = np.linspace(0, len(mono) - 1, n_out)
            mono = np.interp(indices, np.arange(len(mono)), mono)

    return mono.astype(np.float32)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(audio_data: dict,
             top_n: int = 10,
             segment_sec: float = 1.0) -> dict:
    """Run YAMNet classification on audio.

    Args:
        audio_data: dict from src.capture.loader
        top_n: number of top classes to return per segment and overall
        segment_sec: group results into segments of this length (seconds)

    Returns:
        dict with keys:
          - top_classes: list of {class_name, score} dicts (overall top-N)
          - segments: list of {start_sec, end_sec, top_classes} dicts
          - all_class_names: list of YAMNet class names detected anywhere
          - model: "yamnet"
    """
    model, class_names = _load_yamnet()
    waveform = _prepare_audio(audio_data)

    import tensorflow as tf

    # YAMNet returns (scores, embeddings, log_mel_spectrogram)
    scores, embeddings, log_mel = model(waveform)
    scores = scores.numpy()  # (n_frames, 521)

    n_frames = scores.shape[0]
    total_duration = len(waveform) / _YAMNET_SR
    frame_times = np.arange(n_frames) * _YAMNET_FRAME_HOP_SEC

    # --- Overall top-N (mean scores across all frames) ---
    mean_scores = scores.mean(axis=0)
    top_idx = np.argsort(mean_scores)[::-1][:top_n]
    top_classes = [
        {"class_name": class_names[i], "score": round(float(mean_scores[i]), 4)}
        for i in top_idx
    ]

    # --- Segment-level results ---
    segments = []
    seg_start = 0.0
    while seg_start < total_duration:
        seg_end = min(seg_start + segment_sec, total_duration)
        # Find frames that fall within this segment
        mask = (frame_times >= seg_start) & (frame_times < seg_end)
        if mask.any():
            seg_scores = scores[mask].mean(axis=0)
            seg_top_idx = np.argsort(seg_scores)[::-1][:top_n]
            seg_classes = [
                {"class_name": class_names[i], "score": round(float(seg_scores[i]), 4)}
                for i in seg_top_idx
            ]
        else:
            seg_classes = []

        segments.append({
            "start_sec": round(seg_start, 3),
            "end_sec": round(seg_end, 3),
            "top_classes": seg_classes,
        })
        seg_start += segment_sec

    # Collect every class that appeared in any segment's top-N
    all_detected = set()
    for seg in segments:
        for c in seg["top_classes"]:
            all_detected.add(c["class_name"])

    return {
        "top_classes": top_classes,
        "segments": segments,
        "all_class_names": sorted(all_detected),
        "model": "yamnet",
        "n_frames": int(n_frames),
        "duration_sec": round(total_duration, 3),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(audio_data: dict,
        top_n: int = 10,
        segment_sec: float = 1.0) -> dict:
    """Entry point for YAMNet sound classification.

    Falls back gracefully if TensorFlow is not installed — returns empty
    results with a warning instead of raising.

    Args:
        audio_data: dict from src.capture.loader.run()
        top_n: number of top classes per segment / overall
        segment_sec: segment duration in seconds

    Returns:
        Classification result dict (see classify()), or empty-result dict
        if TensorFlow is unavailable.
    """
    try:
        result = classify(audio_data, top_n=top_n, segment_sec=segment_sec)
        _print_summary(result)
        return result
    except ImportError as exc:
        warnings.warn(
            f"YAMNet classification unavailable — {exc}. "
            "Install tensorflow and tensorflow-hub to enable: "
            "pip install tensorflow tensorflow-hub",
            stacklevel=2,
        )
        return _empty_result()
    except Exception as exc:
        warnings.warn(
            f"YAMNet classification failed: {exc}",
            stacklevel=2,
        )
        return _empty_result()


def _empty_result() -> dict:
    """Return a valid but empty classification result."""
    return {
        "top_classes": [],
        "segments": [],
        "all_class_names": [],
        "model": None,
        "n_frames": 0,
        "duration_sec": 0.0,
    }


def _print_summary(result: dict):
    """Print a human-readable summary of classification results."""
    print("\n" + "=" * 60)
    print("YAMNET SOUND CLASSIFICATION")
    print("=" * 60)
    print(f"Frames analysed: {result['n_frames']}  |  "
          f"Duration: {result['duration_sec']:.1f}s  |  "
          f"Segments: {len(result['segments'])}")
    print(f"\nTOP CLASSES (overall):")
    for i, c in enumerate(result["top_classes"], 1):
        bar = "#" * int(c["score"] * 40)
        print(f"  [{i:2d}] {c['score']:.3f}  {bar}  {c['class_name']}")

    if result["segments"]:
        print(f"\nSEGMENT DETAIL (first 3 shown):")
        for seg in result["segments"][:3]:
            label = f"  {seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s:"
            names = ", ".join(c["class_name"] for c in seg["top_classes"][:3])
            print(f"{label}  {names}")
        if len(result["segments"]) > 3:
            print(f"  ... ({len(result['segments']) - 3} more segments)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    from src.capture.loader import run as load_audio

    parser = argparse.ArgumentParser(description="YAMNet Sound Classification")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--segment", type=float, default=1.0,
                        help="Segment length in seconds")
    args = parser.parse_args()

    audio_data = load_audio(args.input)
    result = run(audio_data, top_n=args.top_n, segment_sec=args.segment)

    import json
    print("\n" + json.dumps(result, indent=2))
