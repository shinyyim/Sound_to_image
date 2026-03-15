# Sound to Image — Generating Spatial Environments from Ambisonic Sound Data

## Project Overview
AAS Spring 2025 research project at SCI-Arc. Translates ambisonic spatial audio recordings into visual and spatial environments through a 7-stage AI pipeline.

**Research Question:** Can spatial audio recordings be used as a generative input for constructing visual and spatial world models?

## Architecture — 7-Stage Pipeline

1. **Capture** (`src/capture/`) — Multi-channel FOA loader, 4ch validation, resampling
2. **Spatial Analysis** (`src/spatial_analysis/`) — FOA directional cues, spectrogram, onset density, RT60, band energy, spectral entropy
3. **Audio Embedding** (`src/audio_embedding/`) — LAION-CLAP (HTSAT-tiny, 512-dim) + environment matching against 20 descriptions
4. **Metadata Predictor** (`src/metadata_predictor/`) — Rule-based environment classification, enhanced by CLAP matches + YAMNet (when available)
5. **LLM Interpreter** (`src/llm_interpreter/`) — Claude API: metadata → scene paragraph + diffusion prompts (Kodak Portra 400 suffix)
6. **Image Generation** (`src/image_generation/`) — GPT Image (gpt-image-1), DALL-E 3, Gemini, Stability API, placeholder
7. **3D Reconstruction** (`src/reconstruction_3d/`) — Gaussian Splatting / NeRF (not yet implemented)

## Three Dataset Types

- **Dataset A (Synthetic):** TAU/STARSS/SELD — labeled spatial audio for supervised training
- **Dataset B (Ambisonic):** Real field recordings — environmental grounding
- **Dataset C (Real-World):** Degraded online recordings — forensic stress-test / evaluation only

## Tech Stack
- Python 3.14 (venv at `./venv`)
- PyTorch, librosa, soundfile (audio processing)
- LAION-CLAP (audio-text embeddings)
- Anthropic Claude API (LLM scene interpretation)
- OpenAI GPT Image API (image generation)
- Nerfstudio / gsplat (3D reconstruction — future)

## Commands
```bash
# Activate venv
source venv/bin/activate

# Run full pipeline
python run_pipeline.py --input data/dataset_b_ambisonic/example.wav --image-method gpt-image

# Run analysis only
python analyze_sound.py --input path/to/audio.wav --skip-clap

# Start dashboard server (port 8420)
python server.py

# Download YouTube audio
yt-dlp -f bestaudio -x --audio-format wav -o "data/dataset_b_ambisonic/%(title)s.%(ext)s" "URL"
```

## Server Endpoints
- `GET /dashboard.html` — Main dashboard (or embedded version from outputs/metadata/)
- `POST /api/upload` — Upload audio file → full analysis (librosa + CLAP)
- `POST /api/pipeline` — Run stages 4→5→6 (predict, interpret, generate)
- `POST /api/chat` — Claude API proxy

## Code Conventions
- Module folders: `src/capture/`, `src/spatial_analysis/`, etc. (no number prefixes — Python can't import them)
- Every module exposes a `run()` function as its entry point
- Intermediate outputs go to `outputs/` organized by type
- Pipeline runs saved to `outputs/runs/YYYYMMDD_HHMMSS/`
- All audio I/O uses soundfile; spectral analysis uses librosa
- API keys stored in `.env` (gitignored), loaded via python-dotenv

## Dashboard
- Balenciaga font, monochrome UI (black + white only)
- Real-time visualizations via Web Audio API
- Live stats (onset, DR, entropy, RT60, enclosure, L/M/H)
- Pipeline buttons: [Predict] [Interpret] [Full-Image]
- Custom prompt input for direct image generation
- Image click → fullscreen modal

@AAS_SPRING_2025_Updated_Proposal.md
