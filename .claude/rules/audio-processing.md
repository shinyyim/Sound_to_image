---
paths:
  - "src/01_capture/**/*.py"
  - "src/02_spatial_analysis/**/*.py"
  - "src/03_audio_embedding/**/*.py"
---

# Audio Processing Rules

- Use soundfile for all audio I/O (not scipy.io.wavfile)
- Use librosa for spectral analysis and feature extraction
- All audio loaded as float32 numpy arrays
- Ambisonic FOA files have 4 channels: W (omni), X, Y, Z
- Sample rate default: 24000 Hz (match TAU/STARSS datasets)
- Always validate channel count before processing FOA data
