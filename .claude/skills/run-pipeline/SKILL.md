---
name: run-pipeline
description: Run the full sound-to-image pipeline on an ambisonic audio file
disable-model-invocation: true
---

Run the full 7-stage pipeline on the given audio file:

1. Load the ambisonic audio from $ARGUMENTS
2. Run spatial audio analysis (FOA extraction, source detection)
3. Generate audio embeddings (CLAP/AST)
4. Predict scene metadata (structured JSON)
5. Interpret scene via LLM (scene paragraph + diffusion prompt)
6. Generate multi-view images (front, lateral, aerial)
7. Reconstruct 3D environment (Gaussian Splatting/NeRF)

Save all intermediate outputs to `outputs/` with a timestamped run ID.
Print a summary of each stage's output before proceeding to the next.
