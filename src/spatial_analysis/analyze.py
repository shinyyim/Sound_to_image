"""
Module 02 — Spatial Audio Analysis
Extracts spatial features from FOA ambisonic recordings:
- Spectrogram (mel + STFT)
- FOA directional cues (intensity vectors, DOA estimation)
- Onset density, loudness variance, spectral entropy
- Low/mid/high band energy
- Reverberation estimation (RT60 approximation, DRR)
- Motion trajectory estimation
"""

import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal


# ──────────────────────────────────────────────
# Spectrogram Features
# ──────────────────────────────────────────────

def compute_spectrogram(audio_mono: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> dict:
    """Compute STFT and mel spectrogram for a mono signal."""
    stft = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    mel = librosa.feature.melspectrogram(S=mag**2, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return {
        "stft_magnitude": mag,
        "mel_spectrogram_db": mel_db,
        "n_fft": n_fft,
        "hop_length": hop,
        "freq_bins": mag.shape[0],
        "time_frames": mag.shape[1],
    }


# ──────────────────────────────────────────────
# FOA Directional Cues
# ──────────────────────────────────────────────

def compute_foa_intensity(audio: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> dict:
    """Compute FOA active intensity vectors for direction-of-arrival estimation.

    For FOA (W, X, Y, Z):
    - Intensity_x = Re(W* · X) → front-back
    - Intensity_y = Re(W* · Y) → left-right
    - Intensity_z = Re(W* · Z) → up-down

    Returns azimuth and elevation over time.
    """
    if audio.shape[1] < 4:
        return {"error": "Not FOA — need 4 channels for intensity analysis"}

    w = librosa.stft(audio[:, 0], n_fft=n_fft, hop_length=hop)
    x = librosa.stft(audio[:, 1], n_fft=n_fft, hop_length=hop)
    y = librosa.stft(audio[:, 2], n_fft=n_fft, hop_length=hop)
    z = librosa.stft(audio[:, 3], n_fft=n_fft, hop_length=hop)

    # Active intensity vectors (real part of cross-spectrum)
    ix = np.real(np.conj(w) * x)
    iy = np.real(np.conj(w) * y)
    iz = np.real(np.conj(w) * z)

    # Average across frequency bins for each time frame
    ix_avg = np.mean(ix, axis=0)
    iy_avg = np.mean(iy, axis=0)
    iz_avg = np.mean(iz, axis=0)

    # DOA: azimuth and elevation per time frame
    azimuth = np.arctan2(iy_avg, ix_avg) * (180.0 / np.pi)
    elevation = np.arctan2(iz_avg, np.sqrt(ix_avg**2 + iy_avg**2)) * (180.0 / np.pi)

    # Directional energy (magnitude of intensity vector)
    directional_energy = np.sqrt(ix_avg**2 + iy_avg**2 + iz_avg**2)

    return {
        "azimuth_deg": azimuth,           # shape: (time_frames,)
        "elevation_deg": elevation,       # shape: (time_frames,)
        "directional_energy": directional_energy,
        "mean_azimuth": float(np.mean(azimuth)),
        "mean_elevation": float(np.mean(elevation)),
        "azimuth_variance": float(np.var(azimuth)),
        "elevation_variance": float(np.var(elevation)),
        "dominant_direction": _azimuth_to_label(float(np.median(azimuth))),
    }


def _azimuth_to_label(azimuth_deg: float) -> str:
    """Convert azimuth in degrees to a human-readable direction label."""
    az = azimuth_deg % 360
    if az < 22.5 or az >= 337.5:
        return "front"
    elif az < 67.5:
        return "front-left"
    elif az < 112.5:
        return "left"
    elif az < 157.5:
        return "rear-left"
    elif az < 202.5:
        return "rear"
    elif az < 247.5:
        return "rear-right"
    elif az < 292.5:
        return "right"
    else:
        return "front-right"


# ──────────────────────────────────────────────
# Temporal & Spectral Features
# ──────────────────────────────────────────────

def compute_onset_density(audio_mono: np.ndarray, sr: int) -> dict:
    """Detect onsets and compute event density (onsets per second)."""
    onset_frames = librosa.onset.onset_detect(y=audio_mono, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration = len(audio_mono) / sr
    density = len(onset_frames) / duration if duration > 0 else 0

    return {
        "onset_count": int(len(onset_frames)),
        "onset_density_per_sec": float(density),
        "onset_times": onset_times.tolist(),
        "duration": float(duration),
    }


def compute_loudness_variance(audio_mono: np.ndarray, sr: int, frame_length: int = 2048, hop: int = 512) -> dict:
    """Compute RMS loudness and its variance over time."""
    rms = librosa.feature.rms(y=audio_mono, frame_length=frame_length, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_db_mean": float(np.mean(rms_db)),
        "rms_db_std": float(np.std(rms_db)),
        "dynamic_range_db": float(np.max(rms_db) - np.min(rms_db)),
    }


def compute_spectral_entropy(audio_mono: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> dict:
    """Compute spectral entropy — measures how spread the energy is across frequencies.
    High entropy = noise-like / diffuse.  Low entropy = tonal / focused.
    """
    stft = np.abs(librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop))
    # Normalize each frame to a probability distribution
    power = stft ** 2
    frame_sums = power.sum(axis=0, keepdims=True)
    frame_sums[frame_sums == 0] = 1e-10
    prob = power / frame_sums

    # Shannon entropy per frame
    entropy = -np.sum(prob * np.log2(prob + 1e-10), axis=0)
    max_entropy = np.log2(stft.shape[0])  # max possible entropy

    return {
        "spectral_entropy_mean": float(np.mean(entropy)),
        "spectral_entropy_std": float(np.std(entropy)),
        "spectral_entropy_normalized": float(np.mean(entropy) / max_entropy),
        "interpretation": "diffuse/noisy" if np.mean(entropy) / max_entropy > 0.7 else "tonal/focused",
    }


def compute_band_energy(audio_mono: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> dict:
    """Compute energy in low, mid, and high frequency bands."""
    stft = np.abs(librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = stft ** 2

    # Band boundaries
    low_mask = freqs < 300
    mid_mask = (freqs >= 300) & (freqs < 2000)
    high_mask = freqs >= 2000

    low_energy = float(np.mean(power[low_mask, :])) if low_mask.any() else 0
    mid_energy = float(np.mean(power[mid_mask, :])) if mid_mask.any() else 0
    high_energy = float(np.mean(power[high_mask, :])) if high_mask.any() else 0

    total = low_energy + mid_energy + high_energy + 1e-10

    return {
        "low_energy_ratio": float(low_energy / total),
        "mid_energy_ratio": float(mid_energy / total),
        "high_energy_ratio": float(high_energy / total),
        "low_hz": "0-300",
        "mid_hz": "300-2000",
        "high_hz": "2000+",
    }


# ──────────────────────────────────────────────
# Reverberation Estimation
# ──────────────────────────────────────────────

def estimate_reverberation(audio_mono: np.ndarray, sr: int) -> dict:
    """Estimate reverberation characteristics using autocorrelation decay.

    Approximates:
    - RT60-like decay (how long energy takes to drop 60dB)
    - Direct-to-reverberant ratio (DRR)
    - Enclosure estimation (open vs enclosed)
    """
    # Autocorrelation of the signal
    n = len(audio_mono)
    autocorr = np.correlate(audio_mono[:min(n, sr * 2)], audio_mono[:min(n, sr * 2)], mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Find where autocorrelation drops below thresholds
    decay_10 = np.argmax(autocorr < 0.1) / sr if np.any(autocorr < 0.1) else 0
    decay_01 = np.argmax(autocorr < 0.01) / sr if np.any(autocorr < 0.01) else 0

    # RT60 approximation (extrapolate from T10 or T20)
    rt60_approx = decay_10 * 6 if decay_10 > 0 else 0

    # DRR: ratio of early energy (first 2.5ms) to late energy
    early_samples = int(0.0025 * sr)
    early_energy = np.sum(audio_mono[:early_samples] ** 2)
    late_energy = np.sum(audio_mono[early_samples:] ** 2) + 1e-10
    drr_db = float(10 * np.log10(early_energy / late_energy + 1e-10))

    # Classify enclosure
    if rt60_approx > 1.5:
        enclosure = "highly reverberant / large enclosed"
    elif rt60_approx > 0.6:
        enclosure = "medium reverberant / enclosed"
    elif rt60_approx > 0.2:
        enclosure = "low reverberant / semi-open"
    else:
        enclosure = "dry / open"

    return {
        "rt60_approx_sec": float(rt60_approx),
        "decay_10_sec": float(decay_10),
        "drr_db": drr_db,
        "enclosure_estimate": enclosure,
    }


# ──────────────────────────────────────────────
# Motion Trajectory (from FOA)
# ──────────────────────────────────────────────

def estimate_motion(foa_intensity: dict, sr: int, hop: int = 512) -> dict:
    """Estimate source motion from azimuth/elevation changes over time."""
    if "error" in foa_intensity:
        return {"error": foa_intensity["error"]}

    azimuth = foa_intensity["azimuth_deg"]
    elevation = foa_intensity["elevation_deg"]

    # Angular velocity (degrees per second)
    time_step = hop / sr
    az_velocity = np.diff(azimuth) / time_step
    el_velocity = np.diff(elevation) / time_step

    angular_speed = np.sqrt(az_velocity**2 + el_velocity**2)

    # Classify motion
    mean_speed = float(np.mean(angular_speed))
    if mean_speed > 100:
        motion_type = "rapid movement"
    elif mean_speed > 30:
        motion_type = "moderate movement"
    elif mean_speed > 5:
        motion_type = "slow movement"
    else:
        motion_type = "static"

    return {
        "mean_angular_speed_deg_s": mean_speed,
        "max_angular_speed_deg_s": float(np.max(angular_speed)),
        "motion_type": motion_type,
        "azimuth_range_deg": float(np.ptp(azimuth)),
        "elevation_range_deg": float(np.ptp(elevation)),
    }


# ──────────────────────────────────────────────
# Main Analysis Pipeline
# ──────────────────────────────────────────────

def run(audio_data: dict, output_dir: str = None) -> dict:
    """Run full spatial audio analysis on loaded audio data.

    Args:
        audio_data: Output from src.capture.loader.run()
        output_dir: Optional path to save results JSON

    Returns:
        Complete analysis results dict
    """
    audio = audio_data["audio"]
    sr = audio_data["sr"]
    is_foa = audio_data["is_foa"]

    # Use W channel (omnidirectional) for mono analysis, or channel 0
    mono = audio[:, 0]

    print("Running spatial audio analysis...")

    # 1. Spectrogram
    print("  [1/7] Computing spectrogram...")
    spec = compute_spectrogram(mono, sr)

    # 2. FOA intensity / DOA
    print("  [2/7] Extracting FOA directional cues...")
    if is_foa:
        foa_intensity = compute_foa_intensity(audio, sr)
    else:
        foa_intensity = {"error": "Not FOA — skipping directional analysis"}

    # 3. Onset density
    print("  [3/7] Detecting onsets...")
    onsets = compute_onset_density(mono, sr)

    # 4. Loudness variance
    print("  [4/7] Computing loudness variance...")
    loudness = compute_loudness_variance(mono, sr)

    # 5. Spectral entropy
    print("  [5/7] Computing spectral entropy...")
    entropy = compute_spectral_entropy(mono, sr)

    # 6. Band energy
    print("  [6/7] Computing band energy distribution...")
    bands = compute_band_energy(mono, sr)

    # 7. Reverberation
    print("  [7/7] Estimating reverberation...")
    reverb = estimate_reverberation(mono, sr)

    # Motion (if FOA)
    motion = estimate_motion(foa_intensity, sr) if is_foa else {"error": "Not FOA"}

    # Compile results (only JSON-serializable values)
    results = {
        "file": audio_data["filepath"],
        "duration_sec": audio_data["duration"],
        "sample_rate": sr,
        "channels": audio_data["channels"],
        "is_foa": is_foa,
        "spectrogram": {
            "n_fft": spec["n_fft"],
            "hop_length": spec["hop_length"],
            "freq_bins": spec["freq_bins"],
            "time_frames": spec["time_frames"],
        },
        "foa_directional": {
            k: v for k, v in foa_intensity.items()
            if not isinstance(v, np.ndarray)
        },
        "onset_density": {
            "onset_count": onsets["onset_count"],
            "onset_density_per_sec": onsets["onset_density_per_sec"],
        },
        "loudness": loudness,
        "spectral_entropy": entropy,
        "band_energy": bands,
        "reverberation": reverb,
        "motion": motion,
    }

    # Print summary
    _print_summary(results)

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_data["filepath"]).stem
        out_path = output_dir / f"{stem}_analysis.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved analysis to: {out_path}")

    return results


def _print_summary(results: dict):
    """Print a human-readable summary of analysis results."""
    print("\n" + "=" * 60)
    print("SPATIAL AUDIO ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"File: {results['file']}")
    print(f"Duration: {results['duration_sec']:.2f}s | SR: {results['sample_rate']} Hz | FOA: {results['is_foa']}")

    if "error" not in results["foa_directional"]:
        d = results["foa_directional"]
        print(f"\nDIRECTIONAL:")
        print(f"  Dominant direction: {d.get('dominant_direction', 'N/A')}")
        print(f"  Mean azimuth: {d.get('mean_azimuth', 0):.1f}° | Mean elevation: {d.get('mean_elevation', 0):.1f}°")
        print(f"  Azimuth variance: {d.get('azimuth_variance', 0):.1f}°²")

    o = results["onset_density"]
    print(f"\nEVENT DENSITY:")
    print(f"  Onsets: {o['onset_count']} | Density: {o['onset_density_per_sec']:.2f}/sec")

    l = results["loudness"]
    print(f"\nLOUDNESS:")
    print(f"  Mean RMS: {l['rms_db_mean']:.1f} dB | Dynamic range: {l['dynamic_range_db']:.1f} dB")

    e = results["spectral_entropy"]
    print(f"\nSPECTRAL CHARACTER:")
    print(f"  Entropy: {e['spectral_entropy_normalized']:.3f} ({e['interpretation']})")

    b = results["band_energy"]
    print(f"  Low: {b['low_energy_ratio']*100:.1f}% | Mid: {b['mid_energy_ratio']*100:.1f}% | High: {b['high_energy_ratio']*100:.1f}%")

    r = results["reverberation"]
    print(f"\nREVERBERATION:")
    print(f"  RT60 ≈ {r['rt60_approx_sec']:.2f}s | DRR: {r['drr_db']:.1f} dB")
    print(f"  Enclosure: {r['enclosure_estimate']}")

    if "error" not in results["motion"]:
        m = results["motion"]
        print(f"\nMOTION:")
        print(f"  Type: {m['motion_type']} | Speed: {m['mean_angular_speed_deg_s']:.1f}°/s")

    print("=" * 60)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.capture.loader import run as load_audio

    parser = argparse.ArgumentParser(description="Spatial Audio Analysis")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate")
    args = parser.parse_args()

    audio_data = load_audio(args.input, args.sr)
    run(audio_data, output_dir=args.output)
