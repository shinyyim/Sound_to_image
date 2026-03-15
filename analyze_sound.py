#!/usr/bin/env python3
"""
Sound Analysis Script — Stages 1-3 of the pipeline
Loads ambisonic audio → extracts spatial features → generates CLAP embedding

Usage:
    python analyze_sound.py --input path/to/audio.wav
    python analyze_sound.py --input path/to/audio.wav --skip-clap
    python analyze_sound.py --input path/to/audio.wav --output outputs/metadata
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.capture.loader import run as load_audio
from src.spatial_analysis.analyze import run as analyze_spatial


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ambisonic audio — spatial features + CLAP embedding"
    )
    parser.add_argument("--input", required=True, help="Path to audio file (WAV, FLAC, OGG)")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory for results")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate (default: 24000)")
    parser.add_argument("--skip-clap", action="store_true", help="Skip CLAP embedding (faster)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization dashboard")
    args = parser.parse_args()

    print("=" * 60)
    print("SOUND-TO-IMAGE PIPELINE — AUDIO ANALYSIS")
    print("=" * 60)

    # Stage 1: Load audio
    print("\n[STAGE 1] Loading audio...")
    audio_data = load_audio(args.input, args.sr)

    # Stage 2: Spatial analysis
    print("\n[STAGE 2] Spatial audio analysis...")
    analysis = analyze_spatial(audio_data, output_dir=args.output)

    # Stage 3: CLAP embedding
    if not args.skip_clap:
        print("\n[STAGE 3] CLAP audio embedding...")
        from src.audio_embedding.embed import run as embed_audio
        embedding_result = embed_audio(audio_data, output_dir=args.output)
    else:
        print("\n[STAGE 3] Skipped (--skip-clap)")
        embedding_result = None

    # Visualization
    if not args.no_viz:
        print("\n[VIZ] Generating analysis dashboard...")
        from src.spatial_analysis.visualize import run as visualize
        viz_path = visualize(audio_data, analysis, output_dir=args.output)
    else:
        print("\n[VIZ] Skipped (--no-viz)")
        viz_path = None

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output}/")
    if viz_path:
        print(f"Dashboard: {viz_path}")
    print("=" * 60)

    return analysis, embedding_result


if __name__ == "__main__":
    main()
