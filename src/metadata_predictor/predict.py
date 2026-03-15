"""
Module 04 — Scene Metadata Predictor
Converts spatial audio analysis results into structured scene metadata JSON.

Two modes:
  1. Rule-based (default): Heuristic mapping from audio features → scene metadata.
     Works immediately with no training data required.
  2. ML-based (future): Trained on Dataset A (TAU/STARSS/SELD) for supervised prediction.

Output schema:
{
    "environment_type": str,
    "source_objects": [{"type": str, "direction": str, "motion": str}],
    "acoustic_profile": {"reverberation": str, "density": str, "openness": str},
    "volatility": float,
    "confidence": float
}
"""

import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Environment Classification
# ──────────────────────────────────────────────

def _classify_environment(analysis: dict, classification: dict = None) -> str:
    """Infer environment type from acoustic features, optionally refined by YAMNet.

    If *classification* (output of spatial_analysis.classify.run()) is provided,
    detected YAMNet sound classes are checked first and can override the
    rule-based decision tree when a strong match is found.
    """
    reverb = analysis["reverberation"]
    bands = analysis["band_energy"]
    entropy = analysis["spectral_entropy"]
    onsets = analysis["onset_density"]

    rt60 = reverb["rt60_approx_sec"]
    enclosure = reverb["enclosure_estimate"]
    low_ratio = bands["low_energy_ratio"]
    high_ratio = bands["high_energy_ratio"]
    ent_norm = entropy["spectral_entropy_normalized"]
    density = onsets["onset_density_per_sec"]

    # --- YAMNet-informed override (checked first) --------------------------
    if classification and classification.get("top_classes"):
        yamnet_env = _environment_from_yamnet(classification)
        if yamnet_env is not None:
            return yamnet_env

    # --- Rule-based fallback -----------------------------------------------
    if rt60 > 1.5:
        if density < 1.0:
            return "large interior hall / cathedral"
        else:
            return "large reverberant interior with activity"
    elif rt60 > 0.6:
        if high_ratio > 0.1:
            return "medium room with reflective surfaces"
        elif density > 3.0:
            return "busy enclosed space / transit hub"
        else:
            return "interior corridor or room"
    elif rt60 > 0.2:
        if ent_norm > 0.7:
            return "semi-open urban space"
        elif density > 5.0:
            return "street-level urban environment"
        else:
            return "semi-enclosed threshold or passage"
    else:
        if ent_norm > 0.7 and density > 3.0:
            return "open urban street"
        elif low_ratio > 0.9 and density < 1.0:
            return "natural open landscape"
        elif ent_norm < 0.4:
            return "quiet open environment"
        else:
            return "open outdoor space"


# ──────────────────────────────────────────────
# YAMNet → Environment Mapping
# ──────────────────────────────────────────────

# Mapping from YAMNet class-name keywords to environment labels.
# Checked in priority order; the first group whose keywords ALL appear
# in the detected-class word set wins.
_YAMNET_ENV_RULES = [
    # (required_keywords_set, environment_label)
    ({"bird", "wind"},          "natural landscape"),
    ({"bird", "water"},         "natural landscape near water"),
    ({"ocean", "wave"},         "coastal / seaside environment"),
    ({"rain"},                  "rainy outdoor environment"),
    ({"thunder"},               "stormy outdoor environment"),
    ({"stream", "water"},       "natural landscape near stream"),
    ({"bird"},                  "natural outdoor environment"),
    ({"wind"},                  "open windy landscape"),
    ({"engine", "traffic"},     "roadside urban environment"),
    ({"car", "traffic"},        "roadside urban environment"),
    ({"vehicle"},               "street-level urban environment"),
    ({"siren"},                 "urban emergency / street"),
    ({"speech", "crowd"},       "busy public space with crowd"),
    ({"crowd"},                 "crowded public space"),
    ({"speech", "music"},       "social / entertainment venue"),
    ({"music"},                 "musical venue or space"),
    ({"speech"},                "occupied indoor / social space"),
    ({"dog"},                   "residential or park environment"),
    ({"church", "bell"},        "urban environment near church"),
    ({"insect"},                "natural outdoor environment"),
    ({"typing", "keyboard"},    "office / workstation"),
    ({"mechanical"},            "industrial or mechanical space"),
    ({"construction"},          "construction site"),
]


def _environment_from_yamnet(classification: dict):
    """Derive an environment label from YAMNet top classes.

    Returns None if no rule matched, so the caller falls through to the
    rule-based heuristic.
    """
    detected = set()
    for c in classification.get("top_classes", []):
        for word in c["class_name"].lower().replace(",", " ").split():
            detected.add(word)
        detected.add(c["class_name"].lower())

    for keywords, env_label in _YAMNET_ENV_RULES:
        if all(kw in detected for kw in keywords):
            return env_label

    return None


def _yamnet_agrees_with_rule(yamnet_env: str, rule_env: str) -> bool:
    """Check if YAMNet-derived and rule-based environments broadly agree.

    Simple word-overlap check: if any significant word appears in both
    labels, they are considered in agreement.
    """
    _STOP = {"a", "an", "the", "or", "and", "with", "/"}
    yamnet_words = set(yamnet_env.lower().replace("/", " ").split()) - _STOP
    rule_words = set(rule_env.lower().replace("/", " ").split()) - _STOP
    return bool(yamnet_words & rule_words)


# ──────────────────────────────────────────────
# Source Object Inference
# ──────────────────────────────────────────────

_ONSET_SOURCE_MAP = [
    # (density_range, entropy_range, low_ratio_range, source_type)
    ((0, 0.5), (0, 0.4), (0.8, 1.0), "ambient drone / wind"),
    ((0, 1.0), (0, 0.5), (0.7, 1.0), "natural ambience / rustling"),
    ((0.5, 3.0), (0.3, 0.6), (0.5, 0.9), "footsteps or movement"),
    ((1.0, 5.0), (0.5, 0.8), (0.3, 0.7), "speech or voices"),
    ((3.0, 10.0), (0.6, 0.9), (0.2, 0.6), "crowd murmur"),
    ((0.5, 3.0), (0.4, 0.7), (0.6, 0.9), "mechanical / rhythmic"),
    ((5.0, 20.0), (0.7, 1.0), (0.1, 0.5), "dense urban activity"),
    ((0.1, 2.0), (0.2, 0.5), (0.3, 0.7), "distant traffic"),
    ((2.0, 8.0), (0.5, 0.8), (0.4, 0.8), "mixed environmental sources"),
]


def _infer_sources(analysis: dict, classification: dict = None) -> list:
    """Infer likely sound source objects from audio features.

    When *classification* (YAMNet results) is provided, the top YAMNet
    classes are used directly as source types instead of the rule-based
    onset/entropy heuristic.  The rule-based sources are still appended
    as fallback if YAMNet provides fewer than 4 sources.
    """
    foa = analysis.get("foa_directional", {})
    direction = foa.get("dominant_direction", "omnidirectional")
    motion_info = analysis.get("motion", {})
    motion_type = motion_info.get("motion_type", "static")

    # --- YAMNet-derived sources (preferred) --------------------------------
    yamnet_sources = []
    if classification and classification.get("top_classes"):
        # Use classes with score > threshold as concrete source labels
        _MIN_SCORE = 0.05
        seen = set()
        for c in classification["top_classes"]:
            name = c["class_name"]
            if c["score"] >= _MIN_SCORE and name.lower() not in seen:
                seen.add(name.lower())
                yamnet_sources.append(name)
            if len(yamnet_sources) >= 4:
                break

    # --- Rule-based sources ------------------------------------------------
    onsets = analysis["onset_density"]
    entropy = analysis["spectral_entropy"]
    bands = analysis["band_energy"]

    density = onsets["onset_density_per_sec"]
    ent_norm = entropy["spectral_entropy_normalized"]
    low_ratio = bands["low_energy_ratio"]

    rule_sources = []
    for d_range, e_range, l_range, src_type in _ONSET_SOURCE_MAP:
        if (d_range[0] <= density <= d_range[1] and
            e_range[0] <= ent_norm <= e_range[1] and
            l_range[0] <= low_ratio <= l_range[1]):
            rule_sources.append(src_type)

    if not rule_sources:
        rule_sources = ["unidentified ambient source"]

    # Merge: YAMNet first, then rule-based to fill up to 4 unique sources
    merged = []
    seen_merged = set()
    for s in yamnet_sources + rule_sources:
        key = s.lower()
        if key not in seen_merged:
            seen_merged.add(key)
            merged.append(s)
        if len(merged) >= 4:
            break

    result = []
    for src in merged:
        result.append({
            "type": src,
            "direction": direction,
            "motion": motion_type,
        })

    return result


# ──────────────────────────────────────────────
# Acoustic Profile
# ──────────────────────────────────────────────

def _build_acoustic_profile(analysis: dict) -> dict:
    """Build structured acoustic profile from analysis."""
    reverb = analysis["reverberation"]
    onsets = analysis["onset_density"]
    entropy = analysis["spectral_entropy"]

    rt60 = reverb["rt60_approx_sec"]
    density = onsets["onset_density_per_sec"]
    ent_norm = entropy["spectral_entropy_normalized"]

    # Reverberation level
    if rt60 > 1.5:
        reverb_level = "high"
    elif rt60 > 0.6:
        reverb_level = "medium-high"
    elif rt60 > 0.2:
        reverb_level = "medium"
    elif rt60 > 0.05:
        reverb_level = "low"
    else:
        reverb_level = "very low / dry"

    # Density classification
    if density > 5.0:
        density_level = "high"
    elif density > 2.0:
        density_level = "medium"
    elif density > 0.5:
        density_level = "low"
    else:
        density_level = "very low / sparse"

    # Openness
    enclosure = reverb["enclosure_estimate"]
    if "open" in enclosure:
        openness = "open"
    elif "semi" in enclosure:
        openness = "semi-enclosed"
    else:
        openness = "enclosed"

    return {
        "reverberation": reverb_level,
        "density": density_level,
        "openness": openness,
        "spectral_character": entropy["interpretation"],
    }


# ──────────────────────────────────────────────
# Volatility Index
# ──────────────────────────────────────────────

def _compute_volatility(analysis: dict) -> float:
    """Compute a 0-1 volatility index from audio dynamics.

    High volatility = chaotic, rapidly changing, unstable acoustic field.
    Low volatility = stable, predictable, ambient.
    """
    loudness = analysis["loudness"]
    onsets = analysis["onset_density"]
    entropy = analysis["spectral_entropy"]
    motion = analysis.get("motion", {})

    # Normalized components (each 0-1)
    dynamic_range_norm = min(loudness["dynamic_range_db"] / 80.0, 1.0)
    density_norm = min(onsets["onset_density_per_sec"] / 10.0, 1.0)
    entropy_norm = entropy["spectral_entropy_normalized"]
    rms_var_norm = min(loudness["rms_std"] / 0.2, 1.0)

    motion_norm = 0.0
    if "error" not in motion:
        motion_norm = min(motion.get("mean_angular_speed_deg_s", 0) / 100.0, 1.0)

    volatility = (
        0.25 * dynamic_range_norm +
        0.25 * density_norm +
        0.20 * entropy_norm +
        0.15 * rms_var_norm +
        0.15 * motion_norm
    )

    return round(float(volatility), 4)


# ──────────────────────────────────────────────
# Main Predictor
# ──────────────────────────────────────────────

def _clap_aligns_with_rule(clap_desc: str, rule_env: str) -> bool:
    """Check whether the top CLAP match semantically aligns with the rule-based prediction.

    Uses simple keyword overlap to decide if both methods agree on the
    general environment category.
    """
    # Keyword groups that indicate agreement
    _ALIGNMENT_GROUPS = [
        {"urban", "street", "traffic", "city", "pedestrian", "highway"},
        {"forest", "nature", "birds", "leaves", "field", "wind", "river", "stream", "landscape"},
        {"cathedral", "hall", "interior", "reverberant", "echoing", "room", "corridor"},
        {"subway", "station", "transit", "airport", "terminal", "hub"},
        {"beach", "ocean", "waves", "seagulls"},
        {"rain", "rainy"},
        {"construction", "industrial", "factory"},
        {"library", "office", "quiet"},
        {"park", "children", "playground"},
        {"restaurant", "cafe", "busy"},
        {"parking", "garage", "tunnel", "underground"},
        {"marketplace", "bazaar"},
        {"residential", "neighborhood"},
        {"concert", "performance"},
    ]

    combined_clap = set(clap_desc.lower().split())
    combined_rule = set(rule_env.lower().replace("/", " ").split())

    for group in _ALIGNMENT_GROUPS:
        clap_hit = bool(combined_clap & group)
        rule_hit = bool(combined_rule & group)
        if clap_hit and rule_hit:
            return True
    return False


def run(analysis: dict, embedding: dict = None, output_dir: str = None,
        clap_matches: list = None, classification: dict = None) -> dict:
    """Predict structured scene metadata from spatial analysis results.

    Args:
        analysis: Output from src.spatial_analysis.analyze.run()
        embedding: Optional CLAP embedding result (for future ML model)
        output_dir: Optional path to save metadata JSON
        clap_matches: Optional sorted list of (description, score) from
            src.audio_embedding.match_environment.  When provided the top
            CLAP match can refine the environment_type and the top-3
            matches are included in the output.
        classification: Optional YAMNet classification result from
            src.spatial_analysis.classify.run().  When provided, detected
            sound classes improve environment_type, source_objects, and
            confidence scoring.

    Returns:
        Structured scene metadata dict
    """
    print("Predicting scene metadata...")

    environment = _classify_environment(analysis, classification=classification)
    sources = _infer_sources(analysis, classification=classification)
    profile = _build_acoustic_profile(analysis)
    volatility = _compute_volatility(analysis)

    # Confidence based on data quality
    confidence = 0.7  # base confidence for rule-based
    if analysis["is_foa"]:
        confidence += 0.15  # FOA gives better spatial info
    if embedding is not None:
        confidence += 0.10  # embedding adds semantic grounding
    if analysis["duration_sec"] > 10:
        confidence += 0.05  # longer recordings more reliable

    # ── YAMNet classification refinement ─────────────────
    if classification and classification.get("top_classes"):
        confidence += 0.10  # YAMNet provides concrete class evidence
        # Extra boost when YAMNet environment agrees with rule-based
        rule_env = _classify_environment(analysis, classification=None)
        yamnet_env = _environment_from_yamnet(classification)
        if yamnet_env and _yamnet_agrees_with_rule(yamnet_env, rule_env):
            confidence += 0.05

    # ── CLAP environment refinement ──────────────────────
    clap_top3 = None
    if clap_matches and len(clap_matches) > 0:
        top_desc, top_score = clap_matches[0]

        # If the CLAP top match has a strong score, prefer it as environment_type
        if top_score >= 0.25:
            environment = top_desc

        # Boost confidence when CLAP and rule-based predictions agree
        if _clap_aligns_with_rule(top_desc, _classify_environment(analysis)):
            confidence += 0.05

        # Store top-3 for the output
        clap_top3 = [
            {"description": desc, "similarity": round(float(score), 4)}
            for desc, score in clap_matches[:3]
        ]

    metadata = {
        "environment_type": environment,
        "source_objects": sources,
        "acoustic_profile": profile,
        "volatility": volatility,
        "confidence": round(min(confidence, 1.0), 2),
        "analysis_source": {
            "file": analysis["file"],
            "duration_sec": analysis["duration_sec"],
            "sample_rate": analysis["sample_rate"],
            "is_foa": analysis["is_foa"],
        },
        "raw_features": {
            "rt60_approx": analysis["reverberation"]["rt60_approx_sec"],
            "onset_density": analysis["onset_density"]["onset_density_per_sec"],
            "spectral_entropy_norm": analysis["spectral_entropy"]["spectral_entropy_normalized"],
            "dynamic_range_db": analysis["loudness"]["dynamic_range_db"],
            "low_energy_ratio": analysis["band_energy"]["low_energy_ratio"],
            "mid_energy_ratio": analysis["band_energy"]["mid_energy_ratio"],
            "high_energy_ratio": analysis["band_energy"]["high_energy_ratio"],
        },
    }

    if clap_top3 is not None:
        metadata["clap_environment_matches"] = clap_top3

    if classification and classification.get("top_classes"):
        metadata["yamnet_classification"] = {
            "top_classes": classification["top_classes"],
            "all_detected": classification.get("all_class_names", []),
        }

    _print_metadata(metadata)

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(analysis["file"]).stem
        out_path = output_dir / f"{stem}_scene_metadata.json"
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved scene metadata to: {out_path}")

    return metadata


def _print_metadata(meta: dict):
    """Print human-readable scene metadata summary."""
    print("\n" + "=" * 60)
    print("SCENE METADATA PREDICTION")
    print("=" * 60)
    print(f"Environment: {meta['environment_type']}")
    print(f"Confidence: {meta['confidence']}")
    print(f"Volatility: {meta['volatility']}")

    print(f"\nSOURCE OBJECTS:")
    for i, src in enumerate(meta["source_objects"], 1):
        print(f"  [{i}] {src['type']} — {src['direction']} — {src['motion']}")

    p = meta["acoustic_profile"]
    print(f"\nACOUSTIC PROFILE:")
    print(f"  Reverberation: {p['reverberation']}")
    print(f"  Density: {p['density']}")
    print(f"  Openness: {p['openness']}")
    print(f"  Spectral: {p['spectral_character']}")

    if "yamnet_classification" in meta:
        print(f"\nYAMNET DETECTED CLASSES:")
        for i, c in enumerate(meta["yamnet_classification"]["top_classes"][:5], 1):
            print(f"  [{i}] {c['score']:.3f}  {c['class_name']}")

    if "clap_environment_matches" in meta:
        print(f"\nCLAP ENVIRONMENT MATCHES:")
        for i, m in enumerate(meta["clap_environment_matches"], 1):
            print(f"  [{i}] {m['similarity']:.4f}  {m['description']}")

    print("=" * 60)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scene Metadata Predictor")
    parser.add_argument("--input", required=True, help="Path to analysis JSON file")
    parser.add_argument("--output", default="outputs/metadata", help="Output directory")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        analysis = json.load(f)

    run(analysis, output_dir=args.output)
