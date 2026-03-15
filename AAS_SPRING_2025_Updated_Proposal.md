# AAS SPRING 2025
## Generating Spatial Environments from Ambisonic Sound Data

`SPATIAL AUDIO ANALYSIS` `NEURAL 3D RECONSTRUCTION`

---

## Research Question

> **Can spatial audio recordings be used as a generative input for constructing visual and spatial world models?**

### Sub-Questions

1. How does a machine infer environment from sound?
2. What spatial assumptions emerge from acoustic data?
3. What forms of hallucination appear in AI-generated environments?

---

## Project Overview

This project investigates how spatial audio recordings can function as a generative input for constructing visual and spatial environments through artificial intelligence. Using ambisonic microphones to capture the directional and volumetric qualities of sound fields, the project proposes a modular pipeline that translates acoustic information into visual and spatial models.

The research explores whether machines can reconstruct environments from sound alone — examining how computational systems perform acoustic inference, what spatial assumptions emerge from ambisonic data, and where machine perception hallucinates or invents coherence where only event chaos exists.

---

## Dataset Strategy — Three Epistemic Layers

Rather than training on a single data source, the system uses three dataset types as distinct epistemic layers — each teaching different aspects of the sound-to-space relationship. This structure enables controlled calibration, environmental grounding, and forensic stress-testing within a single modular pipeline.

### Dataset A — Synthetic Spatial Audio
`SUPERVISED CALIBRATION`

**Source:** TAU / STARSS / SELD-style datasets

Controlled, labeled spatial-audio scenes with precise source coordinates, onset/offset times, azimuth, elevation, and motion trajectories. Teaches the system the grammar of sound object + position + movement + environment tag.

**Teaches:**
- Source localization
- Event onset & duration
- Azimuth / elevation / motion
- Acoustic scene structure

---

### Dataset B — Real Ambisonic Field Recordings
`ENVIRONMENTAL GROUNDING`

**Source:** Ambisonic libraries / field archives

Unscripted volumetric recordings of real environments. No clean isolated events — diffuse ambience, directional flow, and environmental texture. Teaches the system how to describe whole environments rather than discrete sound objects.

**Teaches:**
- Reverberation & openness
- Diffuse ambience
- Environmental density
- Spatial atmosphere

---

### Dataset C — Real-World Online Account Recordings
`FORENSIC STRESS TEST`

**Source:** Livestreams, protest footage, CCTV, field documentation

Degraded, compressed, politically charged recordings where signal instability is the subject. Used primarily for evaluation and comparative analysis — not training. Reveals where the model over-aestheticizes, invents coherence, or hallucinates architecture.

**Teaches:**
- Degraded signal behavior
- Acoustic evidence of conflict
- Compression artifact response
- Hallucination analysis

---

### Dataset Role Division

| Dataset Type | Primary Role | Pipeline Stage | What It Teaches |
|---|---|---|---|
| A — Synthetic Spatial | Supervised training | Metadata predictor calibration | Object localization, motion, event logic |
| B — Real Ambisonic | Environmental grounding | LLM interpretation tuning | Environmental texture, reverb, spatial atmosphere |
| C — Real-World Online | Evaluation & adaptation | Forensic stress-test | Ambiguity, instability, evidentiary complexity |

---

## System Architecture — Modular Pipeline

The system is structured as four sequential modules. Each module corresponds to a distinct transformation stage — from raw acoustic capture through spatial evidence extraction, interpretive scene generation, and finally speculative 3D world construction.

```
AMBISONIC RECORDING
        ↓
SPATIAL AUDIO ANALYSIS
        ↓
AUDIO EMBEDDING
        ↓
SCENE METADATA PREDICTOR
        ↓
LLM SCENE INTERPRETATION
        ↓
IMAGE DIFFUSION GENERATION
        ↓
3D RECONSTRUCTION
        ↓
SPATIAL WORLD MODEL
```

### Stage Descriptions

**1. Ambisonic Recording**
Multi-channel FOA capture — directional + volumetric sound field

**2. Spatial Audio Analysis**
FOA directional cue extraction · source detection · motion estimation · reverberation & density profiling

**3. Audio Embedding**
CLAP / Audio Spectrogram Transformer · semantic embedding of acoustic texture · environmental context encoding

**4. Scene Metadata Predictor**
Trained on Dataset A (synthetic) · outputs structured JSON scene data · source type · direction · enclosure · volatility

**5. LLM Scene Interpretation**
Fine-tuned on Datasets A + B · translates metadata → scene paragraph · generates visual diffusion prompt

**6. Image Diffusion Generation**
Front · lateral · aerial viewpoints · multi-view image set for reconstruction

**7. 3D Reconstruction**
Gaussian Splatting or NeRF · speculative spatial world model

---

## Technical Components

### 01 — Spatial Audio Capture

Ambisonic microphones record multi-channel audio preserving directional sound information. First-Order Ambisonics (FOA) output includes an omnidirectional channel and directional channels along X, Y, and Z axes — enabling inference of sound direction, acoustic density, and environmental reverberation.

### 02 — Audio Feature Extraction

From all three dataset types, the system extracts: CLAP or AST embeddings, spectrogram summaries, onset density, loudness variance, spectral entropy, low/mid/high band energy, FOA directional cues, reverberation estimates, and motion trajectory estimates.

### 03 — Scene Metadata Predictor

Trained on Dataset A (synthetic spatial audio). Converts audio features into structured intermediate representations: likely sound sources, directionality, enclosure vs. openness, event density, and volatility index. Output is JSON-formatted scene metadata. This is the system's reality anchor.

**Example output:**
```json
{
  "environment_type": "interior corridor",
  "source_objects": [
    { "type": "footsteps", "direction": "rear-left", "motion": "approaching" },
    { "type": "door", "direction": "front-right", "motion": "static" }
  ],
  "acoustic_profile": {
    "reverberation": "medium-high",
    "density": "low",
    "openness": "enclosed"
  }
}
```

### 04 — LLM Scene Interpretation

Prompt-engineered or lightly fine-tuned using Datasets A and B. Translates structured metadata into scene paragraphs, visual diffusion prompts, and optional confidence statements.

**Example output:**
> *"A semi-enclosed public transit threshold with periodic mechanical arrivals, lateral pass-bys, diffuse crowd murmur, and reflective concrete surfaces."*

### 05 — Image & 3D Generation

Diffusion models generate multi-viewpoint interpretations (front, lateral, aerial) from acoustic scene descriptions. These are converted into spatial environments via Gaussian splatting, NeRF reconstruction, or image-to-3D models — producing speculative architectural spaces derived from acoustic information.

---

## Methodology — Three-Phase Training & Evaluation

Training and evaluation are divided into three phases corresponding to increasing levels of signal complexity and epistemic instability. Each phase uses a different dataset type and produces different research questions about machine spatial inference.

---

### Phase 01 — Controlled Learning `WEEKS 2–3`

**Dataset:** A — Synthetic Spatial Audio
**Goal:** Calibrate the metadata prediction layer on labeled spatial data.

**Key Tasks:**
- Train source detector and direction inference module
- Build acoustic profile estimator (reverberation, density, openness)
- Output structured JSON scene metadata per recording
- Evaluate source localization accuracy against ground-truth labels

**Deliverables:** Calibrated metadata predictor · Sound object metadata · Environmental embedding vectors

---

### Phase 02 — Environmental Grounding `WEEK 4`

**Dataset:** B — Real Ambisonic Field Recordings
**Goal:** Adapt the LLM interpretation layer to whole-environment descriptions.

**Key Tasks:**
- Feed real ambisonic recordings through calibrated metadata layer
- Compare predicted metadata against expected environmental profiles
- Fine-tune LLM prompting for atmosphere, texture, and spatial scale
- Generate multi-view images and assess visual-acoustic coherence

**Deliverables:** Scene descriptions · Visual interpretations · Image dataset for reconstruction

---

### Phase 03 — Forensic Evaluation `WEEKS 5–6`

**Dataset:** C — Real-World Online Account Recordings
**Goal:** Stress-test the pipeline against degraded, politically charged recordings.

**Key Tasks:**
- Run degraded online recordings through full pipeline without retraining
- Document where model overcommits to spatial coherence
- Identify compression artifacts misread as acoustic space
- Analyze hallucination patterns: invented architecture, over-aestheticized conflict

**Deliverables:** Comparative analysis · Hallucination documentation · Evidentiary vs. generative gap study

---

## Production Schedule

### Week 1 — Research and System Design
**Objectives:** Finalize conceptual framework. Define the technical pipeline. Identify software tools. Acquire or test ambisonic microphone. Select dataset sources across all three types.

**Deliverables:** System diagram · Technical stack selection · Initial dataset acquisition plan

---

### Week 2 — Dataset Acquisition and Audio Capture
**Objectives:** Record original ambisonic sound environments. Acquire synthetic spatial audio dataset (TAU/STARSS). Identify real ambisonic field recording sources and online account recording archives.

**Deliverables:** Spatial audio dataset (5–10 environments) · Annotated recordings · Dataset A acquisition complete

---

### Week 3 — Audio Analysis and Metadata Calibration
**Objectives:** Extract spatial features from ambisonic recordings. Train metadata predictor on Dataset A. Generate audio embeddings and build structured scene metadata.

**Deliverables:** Sound object metadata · Environmental embedding vectors · Calibrated metadata predictor

---

### Week 4 — Scene Interpretation and Image Generation
**Objectives:** Adapt LLM interpretation layer using Datasets A and B. Develop prompts translating metadata into scene descriptions. Generate multi-view visual outputs from acoustic environments.

**Deliverables:** Visual interpretations of sound environments · Image dataset for reconstruction

---

### Week 5 — Spatial Reconstruction and Forensic Evaluation
**Objectives:** Convert generated images into spatial models via Gaussian splatting or NeRF. Run Dataset C (real-world recordings) through full pipeline. Document model behavior under degraded input.

**Deliverables:** Reconstructed environments · Point clouds or Gaussian splat scenes · Hallucination analysis report

---

### Week 6 — World Model, Documentation, and Presentation
**Objectives:** Assemble environments into a navigable system. Document the sound-to-world translation process. Produce comparative analysis across all three dataset types.

**Deliverables:** Final environments · System pipeline documentation · Project presentation materials

---

## Expected Outputs

**Working Pipeline**
A modular computational system translating ambisonic recordings into speculative spatial environments through structured metadata, LLM interpretation, and 3D reconstruction.

**Generated World Models**
A series of spatial environments derived from three distinct dataset types — synthetic, ambisonic, and forensic — enabling direct comparative analysis of machine spatial inference.

**Three-Output Evidence Structure**
Each pipeline run produces:
1. Structured evidence layer with spatial metadata
2. Interpretive scene text from LLM
3. Generated spatial world

Speculation is always tethered to prior evidentiary grounding.

**Hallucination Study**
Documentation of where and how the model invents spatial coherence — over-aestheticizing conflict, misreading compression artifacts as acoustic space, hallucinating architecture from event chaos.

**Environments Explored:**
- Urban street soundscape
- Interior architectural space
- Natural landscape
- Degraded online field documentation
- Contested or politically charged acoustic environments

---

## Conceptual Significance

This project explores a shift in architectural representation: rather than generating environments from visual reference material, it proposes sound as a generative input for spatial design. The work examines how machines interpret acoustic environments and how these interpretations produce speculative architectures.

By translating sound fields into spatial models across three epistemic layers — controlled, grounded, and forensic — the project investigates machine perception as a form of environmental imagination, and positions hallucination not as failure but as evidence of the assumptions embedded in machine spatial reasoning.

---

## References

- UT Austin, 2024. *Researchers Use AI to Turn Sound Recordings into Accurate Street Images.* https://news.utexas.edu/2024/11/27/researchers-use-ai-to-turn-sound-recordings-into-accurate-street-images/
- *Nature Humanities and Social Sciences*, 2024. https://www.nature.com/articles/s41599-024-03645-7
- *Computers, Environment and Urban Systems*, ScienceDirect, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0198971524000516
- TAU Spatial Sound Events Dataset (DCASE / Zenodo 3377088). *Sound Event Localization and Detection.*
