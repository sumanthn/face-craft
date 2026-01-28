# FaceCraft Overview

FaceCraft is a **facial uniqueness analyzer** that celebrates what makes each face distinct — not a beauty scoring system.

## Core Analysis Dimensions

| Dimension | What It Measures |
|-----------|------------------|
| **Proportional Signature** | Eye spacing, nose length, lip fullness, forehead, jaw width ratios |
| **Symmetry Profile** | Natural asymmetry patterns, dominant side, individual feature asymmetry |
| **Feature Prominence** | Which features draw the eye first (eyes, brows, cheekbones, etc.) |
| **Geometric Archetype** | Face shape on 2D spectrum (angular↔soft, long↔compact) |
| **Color Character** | Skin undertone (warm/cool/neutral), contrast level, luminance |

## Project Structure

- **`facial_analyzer.py`** — Core analysis engine (~1400 lines)
- **`app.py`** — Gradio web interface (runs on port 8090)
- **`test_celebrities.py`** — Celebrity validation tests
- **`visualize_landmarks.py`** — Landmark debugging/visualization

## Tech Stack

- **InsightFace** with GPU acceleration (106-point landmarks)
- **MediaPipe** as CPU fallback
- **OpenCV** for image processing
- **Gradio** for web UI

## Key Design Principles

1. **No beauty hierarchy** — celebrates distinctiveness, never ranks
2. **Positive framing** — asymmetry = character, not deficiency
3. **Quality-aware** — flags unreliable results from poor images
4. **Explainable** — every output is reasoned and defensible

## Running It

```bash
python app.py  # Web interface at http://0.0.0.0:8090
```

## Future Features (Planned)

- Brand-face matching for casting
- Batch analysis
- Diversity auditing
- REST API
- Campaign consistency analysis
