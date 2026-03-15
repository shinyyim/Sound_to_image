---
name: generate-scene
description: Generate a visual scene and 3D environment from scene metadata JSON
---

Generate visuals from the scene metadata at $ARGUMENTS:

1. Read the JSON metadata file
2. Use LLM interpreter to generate a scene paragraph and diffusion prompt
3. Generate 3 viewpoint images (front, lateral, aerial) via diffusion
4. Run 3D reconstruction on the generated images
5. Save all outputs:
   - Scene text → `outputs/scene_texts/`
   - Images → `outputs/images/`
   - 3D model → `outputs/3d_models/`
6. Print the scene description and paths to all generated files
