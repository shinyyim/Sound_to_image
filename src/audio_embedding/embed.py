"""
Module 03 — Audio Embedding
Generates semantic embeddings using LAION-CLAP.
CLAP maps audio into a shared audio-text embedding space,
enabling direct comparison between sound and language descriptions.
"""

import numpy as np
import torch
from pathlib import Path


def load_clap_model(device: str = None):
    """Load the LAION-CLAP model."""
    import laion_clap

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()  # downloads pretrained weights on first run
    model.to(device)
    model.eval()

    return model, device


def embed_audio(model, audio_data: dict, device: str = "cpu") -> dict:
    """Generate CLAP embedding from audio data.

    Args:
        model: Loaded CLAP model
        audio_data: Output from src.capture.loader.run()
        device: torch device

    Returns:
        dict with embedding vector and metadata
    """
    filepath = audio_data.get("filepath", "")

    # Try file-based embedding first (more reliable)
    if filepath and Path(filepath).exists():
        with torch.no_grad():
            embedding = model.get_audio_embedding_from_filelist(
                x=[filepath],
                use_tensor=False
            )
        embedding = embedding[0]
    else:
        # Fallback to data-based embedding
        audio = audio_data["audio"]
        sr = audio_data["sr"]
        mono = audio[:, 0] if audio.ndim == 2 else audio

        if sr != 48000:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=48000)

        mono = mono.astype(np.float32)

        with torch.no_grad():
            embedding = model.get_audio_embedding_from_data(
                x=[mono],
                use_tensor=False
            )
        embedding = embedding[0]

    return {
        "embedding": embedding,
        "embedding_dim": int(embedding.shape[0]),
        "model": "LAION-CLAP-HTSAT-base",
        "source_file": audio_data["filepath"],
    }


def embed_text(model, text: str, device: str = "cpu") -> np.ndarray:
    """Generate CLAP embedding from a text description.
    Useful for comparing audio embeddings against scene descriptions.
    """
    with torch.no_grad():
        embedding = model.get_text_embedding([text], use_tensor=False)
    return embedding[0]


def compute_similarity(audio_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
    """Cosine similarity between audio and text embeddings."""
    a = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-10)
    b = text_embedding / (np.linalg.norm(text_embedding) + 1e-10)
    return float(np.dot(a, b))


def run(audio_data: dict, output_dir: str = None) -> dict:
    """Entry point for the embedding module.

    Args:
        audio_data: Output from src.capture.loader.run()
        output_dir: Optional path to save embedding

    Returns:
        Embedding results dict
    """
    print("Loading CLAP model...")
    model, device = load_clap_model()

    print("Generating audio embedding...")
    result = embed_audio(model, audio_data, device)

    print(f"  Embedding dim: {result['embedding_dim']}")
    print(f"  Model: {result['model']}")

    # Save embedding
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_data["filepath"]).stem
        npy_path = output_dir / f"{stem}_clap_embedding.npy"
        np.save(str(npy_path), result["embedding"])
        print(f"  Saved embedding to: {npy_path}")

    return result


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.capture.loader import run as load_audio

    parser = argparse.ArgumentParser(description="Audio Embedding (CLAP)")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory")
    args = parser.parse_args()

    audio_data = load_audio(args.input)
    run(audio_data, output_dir=args.output)
