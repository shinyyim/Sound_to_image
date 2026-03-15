"""
Environment Matching via CLAP
Compares audio embeddings against a bank of textual environment descriptions
using CLAP's shared audio-text embedding space. Returns ranked matches
with cosine similarity scores.
"""

import numpy as np

# ──────────────────────────────────────────────
# Environment Description Bank
# ──────────────────────────────────────────────

ENVIRONMENT_DESCRIPTIONS = [
    "dense urban street with traffic and pedestrians",
    "quiet forest with birds and rustling leaves",
    "large cathedral interior with echoing footsteps",
    "subway station with train arrivals and crowd",
    "beach with ocean waves and seagulls",
    "rainy city street",
    "construction site",
    "quiet library or office",
    "park with children playing",
    "industrial factory floor",
    "concert hall during performance",
    "empty parking garage",
    "busy restaurant or cafe",
    "river or stream in nature",
    "airport terminal",
    "residential neighborhood",
    "open field with wind",
    "underground tunnel",
    "marketplace or bazaar",
    "highway overpass",
]


def _embed_descriptions(model, descriptions: list, device: str = "cpu") -> np.ndarray:
    """Embed all environment descriptions in a single batch.

    Returns:
        np.ndarray of shape (N, embed_dim)
    """
    try:
        from src.audio_embedding.embed import embed_text
    except ImportError:
        from src.audio_embedding.embed import embed_text

    embeddings = []
    for desc in descriptions:
        emb = embed_text(model, desc, device=device)
        embeddings.append(emb)
    return np.stack(embeddings, axis=0)


def _cosine_similarities(audio_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one audio embedding and N text embeddings.

    Args:
        audio_emb: shape (D,)
        text_embs: shape (N, D)

    Returns:
        np.ndarray of shape (N,) with similarity scores
    """
    a = audio_emb / (np.linalg.norm(audio_emb) + 1e-10)
    norms = np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-10
    b = text_embs / norms
    return b @ a


def match_from_embedding(
    audio_embedding: np.ndarray,
    model=None,
    device: str = "cpu",
    descriptions: list = None,
) -> list:
    """Lightweight mode: match a pre-computed audio embedding against environment descriptions.

    Args:
        audio_embedding: Pre-computed CLAP embedding vector, shape (D,).
        model: Loaded CLAP model (needed to embed text descriptions).
        device: Torch device string.
        descriptions: Optional custom list of descriptions. Defaults to built-in bank.

    Returns:
        Sorted list of (description, similarity_score) tuples, highest first.
    """
    if model is None:
        try:
            from src.audio_embedding.embed import load_clap_model
        except ImportError:
            from src.audio_embedding.embed import load_clap_model
        model, device = load_clap_model(device)

    if descriptions is None:
        descriptions = ENVIRONMENT_DESCRIPTIONS

    text_embs = _embed_descriptions(model, descriptions, device)
    scores = _cosine_similarities(audio_embedding, text_embs)

    pairs = list(zip(descriptions, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def run(audio_data: dict, descriptions: list = None) -> list:
    """Full pipeline: load model, embed audio, match against environment descriptions.

    Args:
        audio_data: Output from src.capture.loader.run()  (dict with "audio", "sr", etc.)
        descriptions: Optional custom list of descriptions. Defaults to built-in bank.

    Returns:
        Sorted list of (description, similarity_score) tuples, highest first.
    """
    try:
        from src.audio_embedding.embed import load_clap_model, embed_audio
    except ImportError:
        from src.audio_embedding.embed import load_clap_model, embed_audio

    print("Loading CLAP model for environment matching...")
    model, device = load_clap_model()

    print("Embedding audio...")
    emb_result = embed_audio(model, audio_data, device)
    audio_embedding = emb_result["embedding"]

    print("Matching against environment descriptions...")
    matches = match_from_embedding(
        audio_embedding, model=model, device=device, descriptions=descriptions
    )

    # Print top-5
    print("\nTop-5 environment matches:")
    for desc, score in matches[:5]:
        print(f"  {score:.4f}  {desc}")

    return matches


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.capture.loader import run as load_audio

    parser = argparse.ArgumentParser(description="CLAP Environment Matching")
    parser.add_argument("--input", required=True, help="Path to audio file")
    args = parser.parse_args()

    audio_data = load_audio(args.input)
    matches = run(audio_data)

    print("\nAll matches:")
    for desc, score in matches:
        print(f"  {score:.4f}  {desc}")
