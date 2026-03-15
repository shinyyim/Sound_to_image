"""
Module 05 — LLM Scene Interpretation
Translates structured scene metadata into:
  1. Scene paragraph — descriptive text of the inferred environment
  2. Diffusion prompts — optimized for Stable Diffusion / SDXL image generation
     (front, lateral, aerial viewpoints)

Uses Anthropic Claude API for interpretation.
Falls back to template-based generation if API key is not available.
"""

import json
import os
from pathlib import Path


# ──────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a spatial environment interpreter for an architecture research project.
You receive structured JSON metadata derived from ambisonic spatial audio analysis.
Your job is to translate acoustic evidence into vivid spatial descriptions and image generation prompts.

IMPORTANT RULES:
- Ground your descriptions in the acoustic evidence provided. Do not invent details that contradict the data.
- Describe SPACES, not sounds. Translate acoustic properties into architectural and environmental qualities.
- High reverberation → large enclosed volumes, hard reflective surfaces
- Low reverberation / dry → open air, absorptive materials, outdoors
- High onset density → many discrete events, busy environment
- Low onset density → calm, sparse, ambient
- Volatility indicates chaos vs stability of the acoustic field
- The environment_type is an inference — you may refine it but stay consistent with the acoustic profile.
- For diffusion prompts: be specific about materials, lighting, atmosphere, camera angle. No people unless sources suggest crowd/voices.
"""

USER_PROMPT_TEMPLATE = """Analyze this scene metadata derived from spatial audio and produce:

1. **Scene Paragraph**: A 3-4 sentence vivid description of the spatial environment this sound implies. Describe architecture, materials, scale, atmosphere, and lighting. Write as if describing a place, not a sound.

2. **Diffusion Prompts**: Three image generation prompts optimized for Stable Diffusion XL, one for each viewpoint:
   - **Front view**: Eye-level perspective facing into the space
   - **Lateral view**: Side perspective showing depth and spatial extent
   - **Aerial view**: Bird's-eye or elevated view showing layout and scale

Each prompt should be 1-2 sentences, include: environment, materials, lighting, atmosphere, camera angle. ALWAYS end every prompt with exactly this suffix: "shot on 35mm lens, f/2.8, hyper-realistic, raw photo, unedited, Kodak Portra 400, cinematic lighting, 8k, highly detailed, physically based rendering"

## Scene Metadata
```json
{metadata_json}
```

Respond in this exact JSON format:
```json
{{
    "scene_paragraph": "...",
    "prompts": {{
        "front": "...",
        "lateral": "...",
        "aerial": "..."
    }},
    "material_palette": ["material1", "material2", "material3"],
    "lighting_condition": "...",
    "spatial_scale": "intimate / medium / large / vast"
}}
```"""


# ──────────────────────────────────────────────
# Claude API Interpretation
# ──────────────────────────────────────────────

def interpret_with_claude(metadata: dict) -> dict:
    """Use Anthropic Claude API to interpret scene metadata."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    metadata_json = json.dumps(metadata, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.replace("{metadata_json}", metadata_json)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    # Extract text content
    response_text = message.content[0].text

    # Parse JSON from response (handle markdown code blocks)
    json_str = response_text
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    return json.loads(json_str.strip())


# ──────────────────────────────────────────────
# Template-based Fallback
# ──────────────────────────────────────────────

def interpret_with_templates(metadata: dict) -> dict:
    """Template-based interpretation when API is not available."""
    env = metadata["environment_type"]
    profile = metadata["acoustic_profile"]
    sources = metadata["source_objects"]
    volatility = metadata["volatility"]

    # Build scene paragraph from templates
    reverb_desc = {
        "high": "with long, resonant echoes bouncing off distant walls",
        "medium-high": "where sounds linger and reflect off hard surfaces",
        "medium": "with moderate acoustic reflections suggesting partial enclosure",
        "low": "where sounds dissipate quickly into the surrounding space",
        "very low / dry": "in an acoustically dead, open environment",
    }

    density_desc = {
        "high": "A constant stream of activity fills the space.",
        "medium": "Periodic events punctuate the ambient atmosphere.",
        "low": "Sparse, occasional sounds emerge from the stillness.",
        "very low / sparse": "Near silence, with rare disturbances.",
    }

    openness_desc = {
        "open": "The space stretches outward without boundary, open to sky and horizon.",
        "semi-enclosed": "Partial walls or structures create a threshold between interior and exterior.",
        "enclosed": "Walls and ceiling define a contained volume.",
    }

    source_names = [s["type"] for s in sources]
    source_str = ", ".join(source_names) if source_names else "ambient sound"

    paragraph = (
        f"A {env} {reverb_desc.get(profile['reverberation'], '')}. "
        f"{density_desc.get(profile['density'], '')} "
        f"{openness_desc.get(profile['openness'], '')} "
        f"The acoustic signature suggests the presence of {source_str}."
    )

    # Materials
    material_map = {
        "enclosed": ["concrete", "glass", "steel"],
        "semi-enclosed": ["concrete", "wood", "weathered metal"],
        "open": ["earth", "vegetation", "stone"],
    }
    materials = material_map.get(profile["openness"], ["concrete", "stone", "air"])

    # Lighting
    if profile["openness"] == "open":
        lighting = "natural daylight, overcast sky"
    elif profile["openness"] == "semi-enclosed":
        lighting = "mixed natural and artificial light"
    else:
        lighting = "artificial interior lighting, diffused"

    # Scale
    reverb_val = metadata["raw_features"]["rt60_approx"]
    if reverb_val > 1.5:
        scale = "vast"
    elif reverb_val > 0.6:
        scale = "large"
    elif reverb_val > 0.2:
        scale = "medium"
    else:
        scale = "intimate"

    quality = "shot on 35mm lens, f/2.8, hyper-realistic, raw photo, unedited, Kodak Portra 400, cinematic lighting, 8k, highly detailed, physically based rendering"

    prompts = {
        "front": (
            f"Eye-level perspective of a {env}, {materials[0]} and {materials[1]} surfaces, "
            f"{lighting}, atmospheric perspective, {quality}"
        ),
        "lateral": (
            f"Side view of a {env} showing spatial depth, {materials[0]} walls, "
            f"volumetric light, {quality}"
        ),
        "aerial": (
            f"Aerial view looking down at a {env}, showing layout and scale, "
            f"{materials[2]} textures visible, {lighting}, {quality}"
        ),
    }

    return {
        "scene_paragraph": paragraph,
        "prompts": prompts,
        "material_palette": materials,
        "lighting_condition": lighting,
        "spatial_scale": scale,
    }


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def run(metadata: dict, output_dir: str = None, use_api: bool = True) -> dict:
    """Interpret scene metadata into descriptions and diffusion prompts.

    Args:
        metadata: Output from src.metadata_predictor.predict.run()
        output_dir: Optional path to save interpretation
        use_api: If True, try Claude API first; fall back to templates

    Returns:
        Interpretation dict with scene_paragraph, prompts, etc.
    """
    print("Interpreting scene metadata...")

    api_used = False
    if use_api and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            print("  Using Claude API for interpretation...")
            result = interpret_with_claude(metadata)
            api_used = True
        except Exception as e:
            print(f"  Claude API failed ({e}), falling back to templates...")
            result = interpret_with_templates(metadata)
    else:
        if use_api:
            print("  ANTHROPIC_API_KEY not set, using template fallback...")
        result = interpret_with_templates(metadata)

    # Attach metadata reference
    result["interpretation_method"] = "claude_api" if api_used else "template"
    result["source_metadata"] = {
        "environment_type": metadata["environment_type"],
        "volatility": metadata["volatility"],
        "confidence": metadata["confidence"],
    }

    _print_interpretation(result)

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(metadata.get("analysis_source", {}).get("file", "unknown")).stem

        # Save full interpretation JSON
        json_path = output_dir / f"{stem}_interpretation.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved interpretation to: {json_path}")

        # Save scene text separately
        scene_dir = Path(output_dir).parent / "scene_texts"
        scene_dir.mkdir(parents=True, exist_ok=True)
        txt_path = scene_dir / f"{stem}_scene.txt"
        with open(txt_path, "w") as f:
            f.write(result["scene_paragraph"])
        print(f"Saved scene text to: {txt_path}")

        # Save prompts separately for diffusion stage
        prompt_path = output_dir / f"{stem}_prompts.json"
        with open(prompt_path, "w") as f:
            json.dump(result["prompts"], f, indent=2)
        print(f"Saved prompts to: {prompt_path}")

    return result


def _print_interpretation(result: dict):
    """Print interpretation summary."""
    print("\n" + "=" * 60)
    print("SCENE INTERPRETATION")
    print("=" * 60)
    print(f"Method: {result['interpretation_method']}")
    print(f"Scale: {result['spatial_scale']}")
    print(f"Lighting: {result['lighting_condition']}")
    print(f"Materials: {', '.join(result['material_palette'])}")

    print(f"\nSCENE DESCRIPTION:")
    print(f"  {result['scene_paragraph']}")

    print(f"\nDIFFUSION PROMPTS:")
    for view, prompt in result["prompts"].items():
        print(f"  [{view.upper()}] {prompt}")
    print("=" * 60)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Scene Interpretation")
    parser.add_argument("--input", required=True, help="Path to scene metadata JSON")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory")
    parser.add_argument("--no-api", action="store_true", help="Skip API, use templates only")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        metadata = json.load(f)

    run(metadata, output_dir=args.output, use_api=not args.no_api)
