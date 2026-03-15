"""
Module 06 — Image Generation
Generates multi-view images from scene prompts using:
  1. OpenAI DALL-E 3 — via OpenAI API (OPENAI_API_KEY)
  2. Google Gemini Imagen — via Gemini API (GEMINI_API_KEY)
  3. Stable Diffusion XL (local via diffusers) — if GPU available
  4. Stability AI API (stability.ai) — cloud fallback
  5. Placeholder — generates solid-color PNGs for pipeline testing

Produces 3 viewpoints per scene: front, lateral, aerial.
"""

import json
import os
import io
import struct
import zlib
from pathlib import Path
from datetime import datetime


# ──────────────────────────────────────────────
# OpenAI DALL-E 3 Generation
# ──────────────────────────────────────────────

def generate_with_dalle(prompts: dict, output_dir: Path, scene_id: str,
                        size: str = "1024x1024", quality: str = "hd") -> dict:
    """Generate images using OpenAI DALL-E 3 API."""
    import urllib.request
    import urllib.error
    import base64

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    results = {}
    for view, prompt in prompts.items():
        print(f"  Generating {view} view via DALL-E 3...")

        payload = json.dumps({
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "b64_json",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"DALL-E API error {e.code}: {error_body}")

        img_data = response_data["data"][0]
        img_bytes = base64.b64decode(img_data["b64_json"])

        filename = f"{scene_id}_{view}.png"
        filepath = output_dir / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        # Save revised prompt (DALL-E 3 may revise the prompt)
        revised = img_data.get("revised_prompt", prompt)
        prompt_file = output_dir / f"{scene_id}_{view}_revised_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(revised)

        results[view] = str(filepath)
        print(f"    Saved: {filepath}")

    return results


# ──────────────────────────────────────────────
# OpenAI GPT Image Generation (gpt-image-1)
# ──────────────────────────────────────────────

def generate_with_gpt_image(prompts: dict, output_dir: Path, scene_id: str,
                            size: str = "1024x1024", quality: str = "high") -> dict:
    """Generate images using OpenAI gpt-image-1 model."""
    import urllib.request
    import urllib.error
    import base64

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    results = {}
    for view, prompt in prompts.items():
        print(f"  Generating {view} view via gpt-image-1...")

        payload = json.dumps({
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"GPT Image API error {e.code}: {error_body}")

        img_data = response_data["data"][0]
        # gpt-image-1 returns b64_json by default
        if "b64_json" in img_data:
            img_bytes = base64.b64decode(img_data["b64_json"])
        elif "url" in img_data:
            img_bytes = urllib.request.urlopen(img_data["url"]).read()
        else:
            raise RuntimeError("No image data in response")

        filename = f"{scene_id}_{view}.png"
        filepath = output_dir / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        results[view] = str(filepath)
        print(f"    Saved: {filepath}")

    return results


# ──────────────────────────────────────────────
# Google Gemini Imagen Generation
# ──────────────────────────────────────────────

def generate_with_gemini(prompts: dict, output_dir: Path, scene_id: str) -> dict:
    """Generate images using Gemini 2.0 Flash native image generation."""
    import urllib.request
    import urllib.error
    import base64

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

    results = {}
    for view, prompt in prompts.items():
        print(f"  Generating {view} view via Gemini...")

        payload = json.dumps({
            "contents": [{
                "parts": [{"text": f"Generate a photorealistic architectural photograph: {prompt}"}]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }).encode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={api_key}"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Gemini API error {e.code}: {error_body}")

        # Extract image from response parts
        found = False
        candidates = response_data.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if "inlineData" in part:
                    img_bytes = base64.b64decode(part["inlineData"]["data"])
                    filename = f"{scene_id}_{view}.png"
                    filepath = output_dir / filename
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)
                    results[view] = str(filepath)
                    print(f"    Saved: {filepath}")
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"    Warning: No image returned for {view}")

    return results


# ──────────────────────────────────────────────
# Local SDXL Generation (via diffusers)
# ──────────────────────────────────────────────

def generate_with_sdxl(prompts: dict, output_dir: Path, scene_id: str,
                       negative_prompt: str = None, steps: int = 30,
                       guidance_scale: float = 7.5, width: int = 1024, height: int = 1024) -> dict:
    """Generate images using local SDXL model via diffusers."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    print(f"  Loading SDXL pipeline on {device}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_model_cpu_offload()

    if negative_prompt is None:
        negative_prompt = (
            "blurry, low quality, distorted, cartoon, anime, illustration, "
            "text, watermark, logo, people, figures, humans"
        )

    results = {}
    for view, prompt in prompts.items():
        print(f"  Generating {view} view...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]

        filename = f"{scene_id}_{view}.png"
        filepath = output_dir / filename
        image.save(str(filepath))
        results[view] = str(filepath)
        print(f"    Saved: {filepath}")

    return results


# ──────────────────────────────────────────────
# Stability AI API Generation
# ──────────────────────────────────────────────

def generate_with_stability_api(prompts: dict, output_dir: Path, scene_id: str,
                                 width: int = 1024, height: int = 1024,
                                 steps: int = 30, cfg_scale: float = 7.0) -> dict:
    """Generate images using Stability AI REST API."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("STABILITY_API_KEY")
    if not api_key:
        raise ValueError("STABILITY_API_KEY not set")

    negative_prompt = (
        "blurry, low quality, distorted, cartoon, anime, illustration, "
        "text, watermark, logo, people, figures, humans"
    )

    results = {}
    for view, prompt in prompts.items():
        print(f"  Generating {view} view via Stability API...")

        payload = json.dumps({
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": negative_prompt, "weight": -1.0},
            ],
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "steps": steps,
            "samples": 1,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Stability API error {e.code}: {error_body}")

        import base64
        for i, artifact in enumerate(response_data.get("artifacts", [])):
            if artifact.get("finishReason") == "SUCCESS":
                img_bytes = base64.b64decode(artifact["base64"])
                filename = f"{scene_id}_{view}.png"
                filepath = output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                results[view] = str(filepath)
                print(f"    Saved: {filepath}")
                break

    return results


# ──────────────────────────────────────────────
# Placeholder Generation (no dependencies)
# ──────────────────────────────────────────────

def _create_minimal_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Create a minimal valid PNG with a solid color fill. No PIL required."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)

    # IDAT — raw pixel rows with filter byte 0
    raw_rows = b""
    row = bytes([r, g, b]) * width
    for _ in range(height):
        raw_rows += b"\x00" + row
    compressed = zlib.compress(raw_rows)
    idat = chunk(b"IDAT", compressed)

    # IEND
    iend = chunk(b"IEND", b"")

    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def generate_placeholders(prompts: dict, output_dir: Path, scene_id: str,
                          width: int = 1024, height: int = 1024) -> dict:
    """Generate labeled placeholder PNGs for pipeline testing."""
    view_colors = {
        "front": (40, 60, 80),
        "lateral": (60, 40, 80),
        "aerial": (80, 60, 40),
    }

    results = {}
    for view, prompt in prompts.items():
        r, g, b = view_colors.get(view, (50, 50, 50))
        png_bytes = _create_minimal_png(width, height, r, g, b)

        filename = f"{scene_id}_{view}.png"
        filepath = output_dir / filename
        with open(filepath, "wb") as f:
            f.write(png_bytes)

        # Save prompt alongside
        prompt_file = output_dir / f"{scene_id}_{view}_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        results[view] = str(filepath)
        print(f"    Saved placeholder: {filepath}")

    return results


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def run(interpretation: dict, output_dir: str = None, method: str = "auto",
        width: int = 1024, height: int = 1024, steps: int = 30) -> dict:
    """Generate multi-view images from scene interpretation.

    Args:
        interpretation: Output from src.llm_interpreter.interpret.run()
        output_dir: Output directory for images
        method: "dalle" | "gemini" | "sdxl" | "stability_api" | "placeholder" | "auto"
        width: Image width
        height: Image height
        steps: Inference steps (for diffusion)

    Returns:
        dict with generated image paths and metadata
    """
    print("Generating scene images...")

    prompts = interpretation["prompts"]
    source = interpretation.get("source_metadata", {})
    env_type = source.get("environment_type", "unknown")

    # Scene ID for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_id = f"{env_type.replace(' ', '_').replace('/', '_')}_{timestamp}"

    if output_dir is None:
        output_dir = "outputs/images"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose generation method
    if method == "auto":
        # Priority: DALL-E → Gemini → SDXL → Stability API → placeholder
        if os.environ.get("OPENAI_API_KEY"):
            method = "dalle"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            method = "gemini"
        else:
            try:
                import torch
                from diffusers import StableDiffusionXLPipeline  # noqa: F401
                has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
                if has_gpu:
                    method = "sdxl"
                else:
                    method = "stability_api" if os.environ.get("STABILITY_API_KEY") else "placeholder"
            except ImportError:
                method = "stability_api" if os.environ.get("STABILITY_API_KEY") else "placeholder"

    print(f"  Method: {method}")

    if method == "gpt-image":
        size = f"{width}x{height}" if f"{width}x{height}" in ("1024x1024", "1024x1536", "1536x1024") else "1024x1024"
        image_paths = generate_with_gpt_image(prompts, output_dir, scene_id, size=size)
    elif method == "dalle":
        size = f"{width}x{height}" if f"{width}x{height}" in ("1024x1024", "1024x1792", "1792x1024") else "1024x1024"
        image_paths = generate_with_dalle(prompts, output_dir, scene_id, size=size)
    elif method == "gemini":
        image_paths = generate_with_gemini(prompts, output_dir, scene_id)
    elif method == "sdxl":
        image_paths = generate_with_sdxl(prompts, output_dir, scene_id,
                                          width=width, height=height, steps=steps)
    elif method == "stability_api":
        image_paths = generate_with_stability_api(prompts, output_dir, scene_id,
                                                    width=width, height=height, steps=steps)
    elif method == "placeholder":
        image_paths = generate_placeholders(prompts, output_dir, scene_id,
                                             width=width, height=height)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dalle', 'gemini', 'sdxl', 'stability_api', or 'placeholder'")

    result = {
        "scene_id": scene_id,
        "method": method,
        "image_paths": image_paths,
        "prompts": prompts,
        "dimensions": {"width": width, "height": height},
        "environment_type": env_type,
    }

    _print_generation_summary(result)

    # Save generation manifest
    manifest_path = output_dir / f"{scene_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved manifest to: {manifest_path}")

    return result


def _print_generation_summary(result: dict):
    """Print generation summary."""
    print("\n" + "=" * 60)
    print("IMAGE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Scene ID: {result['scene_id']}")
    print(f"Method: {result['method']}")
    print(f"Dimensions: {result['dimensions']['width']}x{result['dimensions']['height']}")

    print(f"\nGENERATED IMAGES:")
    for view, path in result["image_paths"].items():
        print(f"  [{view.upper()}] {path}")
    print("=" * 60)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Generation from Scene Prompts")
    parser.add_argument("--input", required=True, help="Path to interpretation JSON or prompts JSON")
    parser.add_argument("--output", default="outputs/images", help="Output directory")
    parser.add_argument("--method", default="auto", choices=["dalle", "gemini", "sdxl", "stability_api", "placeholder", "auto"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # Handle both interpretation JSON and raw prompts JSON
    if "prompts" not in data:
        data = {"prompts": data, "source_metadata": {}}

    run(data, output_dir=args.output, method=args.method,
        width=args.width, height=args.height, steps=args.steps)
