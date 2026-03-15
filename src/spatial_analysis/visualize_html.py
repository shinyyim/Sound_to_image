"""
Module 02 — Interactive HTML Sound Analysis Dashboard
Reads the dashboard.html template and injects embedded audio + analysis stats.
"""

import json
import base64
import re
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_html_dashboard(audio_data: dict, analysis: dict, output_dir: str = "outputs/metadata") -> str:
    """Generate an interactive HTML dashboard with embedded audio and analysis stats.

    Args:
        audio_data: Output from src.capture.loader.run()
        analysis: Output from src.spatial_analysis.analyze.run()
        output_dir: Where to save the HTML file

    Returns:
        Path to saved HTML file
    """
    filepath = audio_data["filepath"]
    stem = Path(filepath).stem

    # Encode audio as base64
    import soundfile as sf
    import io
    audio = audio_data["audio"]
    sr = audio_data["sr"]

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Read template
    template_path = PROJECT_ROOT / "dashboard.html"
    html = template_path.read_text()

    # Inject the base64 audio
    html = html.replace('const AUDIO_B64 = "";', f'const AUDIO_B64 = "{audio_b64}";')

    # Inject file metadata into header
    html = html.replace(
        'NO FILE LOADED',
        stem.upper()
    )
    html = html.replace(
        '<span class="val" id="metaChannels">—</span>',
        f'<span class="val" id="metaChannels">{audio_data["channels"]}</span>'
    )
    html = html.replace(
        '<span class="val" id="metaSR">—</span>',
        f'<span class="val" id="metaSR">{sr}</span>'
    )

    # Inject analysis stats
    a = analysis
    replacements = {
        'id="statOnset">—': f'id="statOnset">{a["onset_density"]["onset_density_per_sec"]:.2f}',
        'id="statDR">—': f'id="statDR">{a["loudness"]["dynamic_range_db"]:.1f}',
        'id="statEntropy">—': f'id="statEntropy">{a["spectral_entropy"]["spectral_entropy_normalized"]:.3f}',
        'id="statEntropyType">—': f'id="statEntropyType">{a["spectral_entropy"]["interpretation"].upper()}',
        'id="statRT60">—': f'id="statRT60">{a["reverberation"]["rt60_approx_sec"]:.2f}',
        'id="statEnclosure">—': f'id="statEnclosure">{a["reverberation"]["enclosure_estimate"].upper()}',
        'id="statLow">—': f'id="statLow">{a["band_energy"]["low_energy_ratio"]*100:.0f}',
        'id="statMid">—': f'id="statMid">{a["band_energy"]["mid_energy_ratio"]*100:.0f}',
        'id="statHigh">—': f'id="statHigh">{a["band_energy"]["high_energy_ratio"]*100:.0f}',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_dashboard.html"
    out_path.write_text(html)

    print(f"Interactive dashboard saved to: {out_path}")
    return str(out_path)


def run(audio_data: dict, analysis: dict, output_dir: str = "outputs/metadata") -> str:
    """Entry point."""
    return generate_html_dashboard(audio_data, analysis, output_dir)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.capture.loader import run as load_audio
    from src.spatial_analysis.analyze import run as analyze

    parser = argparse.ArgumentParser(description="Generate interactive HTML audio dashboard")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory")
    args = parser.parse_args()

    audio_data = load_audio(args.input)
    analysis = analyze(audio_data)
    run(audio_data, analysis, output_dir=args.output)
