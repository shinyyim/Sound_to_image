# Sound to Image — Generating Spatial Environments from Ambisonic Sound Data

## Project Overview
AAS Spring 2025 research project at SCI-Arc. Translates ambisonic spatial audio recordings into visual and spatial environments through a 7-stage AI pipeline.

**Research Question:** Can spatial audio recordings be used as a generative input for constructing visual and spatial world models?

## Architecture — 7-Stage Pipeline

1. **Capture** (`src/capture/`) — FOA audio loader, 4ch validation, resampling
2. **Spatial Analysis** (`src/spatial_analysis/`) — FOA directional cues, spectrogram, onset, RT60, band energy, entropy
3. **Audio Embedding** (`src/audio_embedding/`) — LAION-CLAP (HTSAT-tiny) + 20-description environment matching
4. **Metadata Predictor** (`src/metadata_predictor/`) — Rule-based + CLAP-enhanced environment classification
5. **LLM Interpreter** (`src/llm_interpreter/`) — Claude API → scene descriptions + diffusion prompts
6. **Image Generation** (`src/image_generation/`) — GPT Image, DALL-E 3, Gemini, Stability API, placeholder
7. **3D Reconstruction** (`src/reconstruction_3d/`) — Not yet implemented

## Commands
```bash
source venv/bin/activate
python server.py                    # Dashboard at localhost:8420
python run_pipeline.py --input <file> --image-method gpt-image
python analyze_sound.py --input <file> --skip-clap
```

## Code Conventions
- Module folders without number prefixes: `src/capture/`, `src/spatial_analysis/`, etc.
- Every module exposes a `run()` function
- Pipeline runs saved to `outputs/runs/YYYYMMDD_HHMMSS/`
- API keys in `.env`, loaded via python-dotenv
- Python 3.14, venv at `./venv`

## Full Proposal
@AAS_SPRING_2025_Updated_Proposal.md
