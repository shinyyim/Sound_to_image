---
paths:
  - "src/06_image_generation/**/*.py"
  - "src/07_3d_reconstruction/**/*.py"
---

# Generation Rules

- Image generation produces 3 viewpoints per scene: front, lateral, aerial
- Output images saved as PNG to outputs/images/ with scene ID prefix
- 3D reconstruction accepts a directory of multi-view images
- Support both Gaussian Splatting and NeRF backends
- 3D outputs saved to outputs/3d_models/
