---
paths:
  - "src/04_metadata_predictor/**/*.py"
  - "src/05_llm_interpreter/**/*.py"
---

# ML Pipeline Rules

- Scene metadata predictor outputs JSON with keys: sources, directions, enclosure, density, volatility
- LLM interpreter receives JSON metadata, outputs scene paragraph + diffusion prompt
- All model configs stored in configs/ as YAML
- Checkpoints saved to models/checkpoints/ with timestamp prefix
- Use torch.save/torch.load for PyTorch checkpoints
