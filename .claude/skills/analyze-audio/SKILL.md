---
name: analyze-audio
description: Analyze an ambisonic audio file and extract spatial features, embeddings, and scene metadata
---

Analyze the ambisonic audio file at $ARGUMENTS:

1. Load the FOA audio (validate 4 channels: W, X, Y, Z)
2. Extract spatial features:
   - Spectrogram summary
   - Onset density and loudness variance
   - Spectral entropy
   - Low/mid/high band energy
   - FOA directional cues
   - Reverberation estimates
3. Generate CLAP or AST embedding
4. Run scene metadata predictor → output JSON
5. Save results to `outputs/metadata/`
6. Print a human-readable summary of the analysis
