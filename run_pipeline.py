#!/usr/bin/env python3
"""
Sound-to-Image Pipeline — Stages 1-6
Loads ambisonic audio → spatial analysis → embedding → metadata prediction →
LLM scene interpretation → image generation

Usage:
    python run_pipeline.py --input data/dataset_b_ambisonic/example.wav
    python run_pipeline.py --input data/dataset_b_ambisonic/example.wav --skip-clap
    python run_pipeline.py --input data/dataset_b_ambisonic/example.wav --no-api --image-method placeholder
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Load .env file
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from src.capture.loader import run as load_audio
from src.spatial_analysis.analyze import run as analyze_spatial
from src.metadata_predictor.predict import run as predict_metadata
from src.llm_interpreter.interpret import run as interpret_scene
from src.image_generation.generate import run as generate_images


def main():
    parser = argparse.ArgumentParser(
        description="Sound-to-Image Pipeline — Stages 1 through 6"
    )
    parser.add_argument("--input", required=True, help="Path to audio file (WAV, FLAC, OGG)")
    parser.add_argument("--output", default="outputs", help="Base output directory")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate (default: 24000)")
    parser.add_argument("--skip-clap", action="store_true", help="Skip CLAP embedding (faster)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization dashboard")
    parser.add_argument("--no-api", action="store_true", help="Skip Claude API, use template interpretation")
    parser.add_argument("--image-method", default="auto",
                        choices=["dalle", "gemini", "sdxl", "stability_api", "placeholder", "auto"],
                        help="Image generation method")
    parser.add_argument("--image-width", type=int, default=1024, help="Generated image width")
    parser.add_argument("--image-height", type=int, default=1024, help="Generated image height")
    parser.add_argument("--image-steps", type=int, default=30, help="Diffusion inference steps")
    args = parser.parse_args()

    output_base = Path(args.output)
    metadata_dir = str(output_base / "metadata")
    images_dir = str(output_base / "images")

    print("=" * 60)
    print("SOUND-TO-IMAGE PIPELINE")
    print("Stages 1-6: Audio → Analysis → Embedding → Metadata → Interpretation → Images")
    print("=" * 60)

    # ── Stage 1: Load audio ──
    print("\n[STAGE 1] Loading audio...")
    audio_data = load_audio(args.input, args.sr)

    # ── Stage 2: Spatial analysis ──
    print("\n[STAGE 2] Spatial audio analysis...")
    analysis = analyze_spatial(audio_data, output_dir=metadata_dir)

    # ── Stage 3: CLAP embedding ──
    embedding_result = None
    if not args.skip_clap:
        print("\n[STAGE 3] CLAP audio embedding...")
        from src.audio_embedding.embed import run as embed_audio
        embedding_result = embed_audio(audio_data, output_dir=metadata_dir)
    else:
        print("\n[STAGE 3] Skipped (--skip-clap)")

    # ── Visualization (optional) ──
    if not args.no_viz:
        print("\n[VIZ] Generating analysis dashboard...")
        from src.spatial_analysis.visualize import run as visualize
        visualize(audio_data, analysis, output_dir=metadata_dir)

    # ── Stage 4: Metadata prediction ──
    print("\n[STAGE 4] Predicting scene metadata...")
    metadata = predict_metadata(analysis, embedding=embedding_result, output_dir=metadata_dir)

    # ── Stage 5: LLM interpretation ──
    print("\n[STAGE 5] Interpreting scene...")
    interpretation = interpret_scene(metadata, output_dir=metadata_dir, use_api=not args.no_api)

    # ── Stage 6: Image generation ──
    print("\n[STAGE 6] Generating images...")
    images = generate_images(
        interpretation,
        output_dir=images_dir,
        method=args.image_method,
        width=args.image_width,
        height=args.image_height,
        steps=args.image_steps,
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Environment: {metadata['environment_type']}")
    print(f"Volatility: {metadata['volatility']} | Confidence: {metadata['confidence']}")
    print(f"Interpretation: {interpretation['interpretation_method']}")
    print(f"Image method: {images['method']}")
    print(f"\nOutputs:")
    print(f"  Metadata:  {metadata_dir}/")
    print(f"  Images:    {images_dir}/")
    for view, path in images["image_paths"].items():
        print(f"    [{view}] {path}")
    print("=" * 60)

    return {
        "analysis": analysis,
        "embedding": embedding_result,
        "metadata": metadata,
        "interpretation": interpretation,
        "images": images,
    }


if __name__ == "__main__":
    main()
