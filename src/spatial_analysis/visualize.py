"""
Module 02 — Spatial Audio Analysis Visualization
Generates a comprehensive visual dashboard of all spatial audio features.
Produces a single multi-panel figure + interactive HTML report.
"""

import json
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import datetime


# ──────────────────────────────────────────────
# Color Palette
# ──────────────────────────────────────────────

COLORS = {
    "bg": "#0d0d1a",
    "panel": "#151528",
    "grid": "#1e1e3a",
    "text": "#e0e0e0",
    "dim": "#666680",
    "accent1": "#00d4ff",   # cyan
    "accent2": "#ff6b6b",   # coral
    "accent3": "#ffd93d",   # gold
    "accent4": "#6bff6b",   # green
    "low": "#ff6b6b",
    "mid": "#ffd93d",
    "high": "#00d4ff",
    "W": "#00d4ff",
    "X": "#ff6b6b",
    "Y": "#ffd93d",
    "Z": "#6bff6b",
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply dark theme styling to an axis."""
    ax.set_facecolor(COLORS["panel"])
    ax.set_title(title, color=COLORS["text"], fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=COLORS["dim"], fontsize=8)
    ax.set_ylabel(ylabel, color=COLORS["dim"], fontsize=8)
    ax.tick_params(colors=COLORS["dim"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])


# ──────────────────────────────────────────────
# Individual Plot Functions
# ──────────────────────────────────────────────

def plot_waveforms(ax, audio: np.ndarray, sr: int, channel_labels: list):
    """Plot all FOA channel waveforms stacked."""
    n_channels = audio.shape[1]
    time = np.arange(audio.shape[0]) / sr

    for i in range(min(n_channels, 4)):
        label = channel_labels[i] if i < len(channel_labels) else f"Ch{i}"
        offset = i * 1.2
        normalized = audio[:, i] / (np.max(np.abs(audio[:, i])) + 1e-10) * 0.5
        ax.plot(time, normalized + offset, color=COLORS[label], linewidth=0.3, alpha=0.8)
        ax.text(-0.02 * time[-1], offset, label, color=COLORS[label],
                fontsize=9, fontweight="bold", ha="right", va="center")

    ax.set_ylim(-0.7, n_channels * 1.2 - 0.5)
    ax.set_yticks([])
    _style_ax(ax, "FOA WAVEFORMS", "Time (s)")


def plot_mel_spectrogram(ax, audio_mono: np.ndarray, sr: int):
    """Plot mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=audio_mono, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, len(audio_mono) / sr, 0, sr / 2])
    _style_ax(ax, "MEL SPECTROGRAM", "Time (s)", "Frequency (Hz)")

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, pad=0.02, aspect=30)
    cbar.ax.tick_params(colors=COLORS["dim"], labelsize=6)
    cbar.set_label("dB", color=COLORS["dim"], fontsize=7)


def plot_foa_doa(ax, azimuth: np.ndarray, elevation: np.ndarray, hop: int, sr: int):
    """Plot direction of arrival (azimuth + elevation) over time."""
    time = np.arange(len(azimuth)) * hop / sr

    ax.scatter(time, azimuth, s=0.5, alpha=0.4, color=COLORS["accent1"], label="Azimuth")
    ax.scatter(time, elevation, s=0.5, alpha=0.4, color=COLORS["accent2"], label="Elevation")

    # Running average
    win = min(50, len(azimuth) // 4)
    if win > 1:
        az_smooth = np.convolve(azimuth, np.ones(win) / win, mode="same")
        el_smooth = np.convolve(elevation, np.ones(win) / win, mode="same")
        ax.plot(time, az_smooth, color=COLORS["accent1"], linewidth=1.5, alpha=0.9)
        ax.plot(time, el_smooth, color=COLORS["accent2"], linewidth=1.5, alpha=0.9)

    ax.axhline(0, color=COLORS["grid"], linewidth=0.5)
    leg = ax.legend(fontsize=7, loc="upper right", framealpha=0.3)
    for text in leg.get_texts():
        text.set_color(COLORS["text"])
    _style_ax(ax, "DIRECTION OF ARRIVAL", "Time (s)", "Degrees")


def plot_polar_doa(ax, azimuth: np.ndarray, directional_energy: np.ndarray):
    """Plot DOA as a polar histogram — where sound is coming from."""
    ax.set_facecolor(COLORS["panel"])

    # Convert to radians
    az_rad = np.deg2rad(azimuth)

    # Weighted histogram
    n_bins = 36
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(az_rad, bins=bins, weights=directional_energy)
    hist = hist / (hist.max() + 1e-10)

    # Plot bars
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 2 * np.pi / n_bins
    bars = ax.bar(bin_centers, hist, width=width, alpha=0.7, color=COLORS["accent1"],
                  edgecolor=COLORS["accent1"], linewidth=0.5)

    # Color by intensity
    for bar, h in zip(bars, hist):
        bar.set_alpha(0.3 + 0.7 * h)

    ax.set_theta_zero_location("N")  # Front = top
    ax.set_theta_direction(-1)       # Clockwise
    ax.set_rticks([])
    ax.tick_params(colors=COLORS["dim"], labelsize=7)
    ax.set_title("SPATIAL DISTRIBUTION", color=COLORS["text"], fontsize=10, fontweight="bold", pad=15)

    # Direction labels
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      ["Front", "FL", "Left", "RL", "Rear", "RR", "Right", "FR"],
                      fontsize=7)


def plot_onset_density(ax, audio_mono: np.ndarray, sr: int):
    """Plot onset strength over time with detected onsets."""
    onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(y=audio_mono, sr=sr, hop_length=512, units="frames")
    time = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512)

    ax.plot(time, onset_env, color=COLORS["accent3"], linewidth=0.5, alpha=0.7)
    ax.fill_between(time, onset_env, alpha=0.15, color=COLORS["accent3"])

    if len(onset_frames) > 0:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        for t in onset_times:
            ax.axvline(t, color=COLORS["accent2"], linewidth=0.3, alpha=0.5)

    density = len(onset_frames) / (len(audio_mono) / sr)
    ax.text(0.98, 0.95, f"{density:.1f} onsets/sec", transform=ax.transAxes,
            color=COLORS["accent3"], fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["panel"], edgecolor=COLORS["accent3"], alpha=0.8))

    _style_ax(ax, "ONSET DENSITY", "Time (s)", "Onset Strength")


def plot_band_energy(ax, audio_mono: np.ndarray, sr: int):
    """Plot energy distribution across low/mid/high bands over time."""
    n_fft = 2048
    hop = 512
    stft = np.abs(librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = stft ** 2
    time = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=hop)

    low_mask = freqs < 300
    mid_mask = (freqs >= 300) & (freqs < 2000)
    high_mask = freqs >= 2000

    low = np.mean(power[low_mask, :], axis=0)
    mid = np.mean(power[mid_mask, :], axis=0)
    high = np.mean(power[high_mask, :], axis=0)

    # Normalize for stacking
    total = low + mid + high + 1e-10
    low_n = low / total
    mid_n = mid / total
    high_n = high / total

    ax.fill_between(time, 0, low_n, color=COLORS["low"], alpha=0.6, label="Low (<300Hz)")
    ax.fill_between(time, low_n, low_n + mid_n, color=COLORS["mid"], alpha=0.6, label="Mid (300-2kHz)")
    ax.fill_between(time, low_n + mid_n, 1, color=COLORS["high"], alpha=0.6, label="High (2kHz+)")

    leg = ax.legend(fontsize=6, loc="upper right", framealpha=0.3)
    for text in leg.get_texts():
        text.set_color(COLORS["text"])
    ax.set_ylim(0, 1)
    _style_ax(ax, "BAND ENERGY DISTRIBUTION", "Time (s)", "Ratio")


def plot_loudness(ax, audio_mono: np.ndarray, sr: int):
    """Plot RMS loudness over time."""
    rms = librosa.feature.rms(y=audio_mono, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    ax.plot(time, rms_db, color=COLORS["accent4"], linewidth=0.8)
    ax.fill_between(time, rms_db, np.min(rms_db), alpha=0.15, color=COLORS["accent4"])

    dynamic_range = np.max(rms_db) - np.min(rms_db)
    ax.text(0.98, 0.95, f"DR: {dynamic_range:.1f} dB", transform=ax.transAxes,
            color=COLORS["accent4"], fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["panel"], edgecolor=COLORS["accent4"], alpha=0.8))

    _style_ax(ax, "LOUDNESS (RMS)", "Time (s)", "dB")


def plot_spectral_entropy(ax, audio_mono: np.ndarray, sr: int):
    """Plot spectral entropy over time."""
    n_fft = 2048
    hop = 512
    stft = np.abs(librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop))
    power = stft ** 2
    frame_sums = power.sum(axis=0, keepdims=True)
    frame_sums[frame_sums == 0] = 1e-10
    prob = power / frame_sums
    entropy = -np.sum(prob * np.log2(prob + 1e-10), axis=0)
    max_entropy = np.log2(stft.shape[0])
    entropy_norm = entropy / max_entropy

    time = librosa.frames_to_time(np.arange(len(entropy_norm)), sr=sr, hop_length=hop)

    ax.plot(time, entropy_norm, color=COLORS["accent1"], linewidth=0.8)
    ax.fill_between(time, entropy_norm, alpha=0.15, color=COLORS["accent1"])
    ax.axhline(0.7, color=COLORS["accent2"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(time[-1] * 0.98, 0.72, "diffuse threshold", color=COLORS["accent2"],
            fontsize=6, ha="right", alpha=0.7)

    mean_ent = float(np.mean(entropy_norm))
    label = "DIFFUSE" if mean_ent > 0.7 else "TONAL"
    ax.text(0.98, 0.95, f"{label} ({mean_ent:.3f})", transform=ax.transAxes,
            color=COLORS["accent1"], fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["panel"], edgecolor=COLORS["accent1"], alpha=0.8))

    ax.set_ylim(0, 1)
    _style_ax(ax, "SPECTRAL ENTROPY", "Time (s)", "Normalized Entropy")


def plot_reverb_summary(ax, reverb: dict, bands: dict):
    """Plot reverberation and acoustic profile as a summary panel."""
    ax.set_facecolor(COLORS["panel"])
    ax.axis("off")

    lines = [
        ("REVERBERATION", COLORS["text"], 12, "bold"),
        ("", COLORS["text"], 6, "normal"),
        (f"RT60 ≈ {reverb['rt60_approx_sec']:.2f} sec", COLORS["accent1"], 11, "bold"),
        (f"DRR: {reverb['drr_db']:.1f} dB", COLORS["accent3"], 10, "normal"),
        (f"Enclosure: {reverb['enclosure_estimate']}", COLORS["accent2"], 10, "normal"),
        ("", COLORS["text"], 6, "normal"),
        ("FREQUENCY BALANCE", COLORS["text"], 12, "bold"),
        ("", COLORS["text"], 6, "normal"),
        (f"Low  (<300Hz):   {bands['low_energy_ratio']*100:5.1f}%", COLORS["low"], 10, "normal"),
        (f"Mid  (300-2kHz): {bands['mid_energy_ratio']*100:5.1f}%", COLORS["mid"], 10, "normal"),
        (f"High (2kHz+):    {bands['high_energy_ratio']*100:5.1f}%", COLORS["high"], 10, "normal"),
    ]

    y = 0.92
    for text, color, size, weight in lines:
        ax.text(0.08, y, text, color=color, fontsize=size, fontweight=weight,
                transform=ax.transAxes, va="top", fontfamily="monospace")
        y -= 0.085

    ax.set_title("ACOUSTIC PROFILE", color=COLORS["text"], fontsize=10, fontweight="bold", pad=8)


# ──────────────────────────────────────────────
# Main Visualization
# ──────────────────────────────────────────────

def generate_dashboard(audio_data: dict, analysis: dict, output_dir: str = "outputs/metadata") -> str:
    """Generate a full analysis dashboard as a PNG image.

    Args:
        audio_data: Output from src.capture.loader.run()
        analysis: Output from src.spatial_analysis.analyze.run()
        output_dir: Where to save the image

    Returns:
        Path to saved image
    """
    audio = audio_data["audio"]
    sr = audio_data["sr"]
    is_foa = audio_data["is_foa"]
    mono = audio[:, 0]

    fig = plt.figure(figsize=(24, 16), facecolor=COLORS["bg"])

    # Title
    stem = Path(audio_data["filepath"]).stem
    fig.suptitle(
        f"SPATIAL AUDIO ANALYSIS — {stem}",
        color=COLORS["text"], fontsize=16, fontweight="bold", y=0.98,
        fontfamily="monospace"
    )
    subtitle = (f"Duration: {audio_data['duration']:.2f}s  |  SR: {sr} Hz  |  "
                f"Channels: {audio_data['channels']}  |  FOA: {is_foa}")
    fig.text(0.5, 0.955, subtitle, ha="center", color=COLORS["dim"], fontsize=9, fontfamily="monospace")

    # Layout: 4 rows x 3 columns with a polar plot
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.05, right=0.95, top=0.93, bottom=0.04)

    # Row 1: Waveforms (wide) + Polar DOA
    ax_wave = fig.add_subplot(gs[0, 0:2])
    plot_waveforms(ax_wave, audio, sr, audio_data["channel_labels"])

    if is_foa and "error" not in analysis.get("foa_directional", {}):
        from src.spatial_analysis.analyze import compute_foa_intensity
        foa = compute_foa_intensity(audio, sr)
        ax_polar = fig.add_subplot(gs[0, 2], projection="polar")
        plot_polar_doa(ax_polar, foa["azimuth_deg"], foa["directional_energy"])
    else:
        ax_info = fig.add_subplot(gs[0, 2])
        plot_reverb_summary(ax_info, analysis["reverberation"], analysis["band_energy"])

    # Row 2: Mel Spectrogram (wide) + DOA time series
    ax_mel = fig.add_subplot(gs[1, 0:2])
    plot_mel_spectrogram(ax_mel, mono, sr)

    if is_foa and "error" not in analysis.get("foa_directional", {}):
        ax_doa = fig.add_subplot(gs[1, 2])
        plot_foa_doa(ax_doa, foa["azimuth_deg"], foa["elevation_deg"], 512, sr)
    else:
        ax_doa = fig.add_subplot(gs[1, 2])
        plot_loudness(ax_doa, mono, sr)

    # Row 3: Onset Density + Band Energy + Spectral Entropy
    ax_onset = fig.add_subplot(gs[2, 0])
    plot_onset_density(ax_onset, mono, sr)

    ax_bands = fig.add_subplot(gs[2, 1])
    plot_band_energy(ax_bands, mono, sr)

    ax_entropy = fig.add_subplot(gs[2, 2])
    plot_spectral_entropy(ax_entropy, mono, sr)

    # Row 4: Loudness + Reverb Summary + Motion/Info
    ax_loud = fig.add_subplot(gs[3, 0])
    plot_loudness(ax_loud, mono, sr)

    if is_foa and "error" not in analysis.get("foa_directional", {}):
        ax_reverb = fig.add_subplot(gs[3, 1])
        plot_reverb_summary(ax_reverb, analysis["reverberation"], analysis["band_energy"])

        ax_motion = fig.add_subplot(gs[3, 2])
        ax_motion.set_facecolor(COLORS["panel"])
        ax_motion.axis("off")
        m = analysis.get("motion", {})
        lines = [
            ("MOTION ANALYSIS", COLORS["text"], 12, "bold"),
            ("", COLORS["text"], 6, "normal"),
            (f"Type: {m.get('motion_type', 'N/A')}", COLORS["accent4"], 11, "bold"),
            (f"Mean speed: {m.get('mean_angular_speed_deg_s', 0):.1f}°/s", COLORS["accent1"], 10, "normal"),
            (f"Max speed: {m.get('max_angular_speed_deg_s', 0):.1f}°/s", COLORS["accent3"], 10, "normal"),
            (f"Azimuth range: {m.get('azimuth_range_deg', 0):.1f}°", COLORS["accent2"], 10, "normal"),
            (f"Elevation range: {m.get('elevation_range_deg', 0):.1f}°", COLORS["dim"], 10, "normal"),
        ]
        y = 0.92
        for text, color, size, weight in lines:
            ax_motion.text(0.08, y, text, color=color, fontsize=size, fontweight=weight,
                          transform=ax_motion.transAxes, va="top", fontfamily="monospace")
            y -= 0.11
        ax_motion.set_title("SOURCE MOTION", color=COLORS["text"], fontsize=10, fontweight="bold", pad=8)
    else:
        ax_reverb = fig.add_subplot(gs[3, 1:3])
        plot_reverb_summary(ax_reverb, analysis["reverberation"], analysis["band_energy"])

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{stem}_analysis_{timestamp}.png"
    fig.savefig(str(out_path), dpi=150, facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)

    print(f"\nDashboard saved to: {out_path}")
    return str(out_path)


def run(audio_data: dict, analysis: dict, output_dir: str = "outputs/metadata") -> str:
    """Entry point for visualization module."""
    print("Generating analysis dashboard...")
    return generate_dashboard(audio_data, analysis, output_dir)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.capture.loader import run as load_audio
    from src.spatial_analysis.analyze import run as analyze

    parser = argparse.ArgumentParser(description="Visualize spatial audio analysis")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory")
    args = parser.parse_args()

    audio_data = load_audio(args.input)
    analysis = analyze(audio_data)
    run(audio_data, analysis, output_dir=args.output)
