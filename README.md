# FaceCraft

**Facial Uniqueness Analyzer** - Celebrates what makes each face distinctly itself.

FaceCraft analyzes facial features and describes what makes a face unique, without scoring beauty. Every face is a composition - we're not judging the art, we're describing what the artist did.

## Features

- **Proportional Signature** - Eye spacing, nose length, lip fullness, jaw width, face ratios
- **Symmetry Character** - Asymmetry patterns (not deficiencies - they add character!)
- **Feature Prominence** - What draws the eye first
- **Geometric Archetype** - Angular/soft and long/compact classification
- **Color Character** - Undertone, contrast level, skin luminance
- **Image Quality Validation** - Warns when photo angle/lighting may affect accuracy

## Tech Stack

- **InsightFace** with ONNX Runtime GPU for fast face detection and 106-point landmarks
- **OpenCV** for image processing
- **Gradio** for web interface
- Runs on NVIDIA GPUs (tested on A40)

## Installation

```bash
pip install insightface onnxruntime-gpu opencv-python gradio numpy
```

## Usage

### Web Interface

```bash
python app.py
```

Opens Gradio interface on port 8090.

### Python API

```python
from facial_analyzer import FacialAnalyzer
import cv2

analyzer = FacialAnalyzer()
image = cv2.imread("photo.jpg")
fingerprint = analyzer.analyze(image)

# Check quality first
if not fingerprint.is_reliable:
    print("Warning: Image quality issues detected")
    for w in fingerprint.quality_warnings:
        print(f"  - {w.message}")

# Get results
print(fingerprint.headline)
print(fingerprint.detailed_report)
```

## Image Quality Checks

FaceCraft validates images before analysis and warns about:

| Issue | Severity | What it means |
|-------|----------|---------------|
| Face too small | Unreliable | Face < 3% of image |
| Not frontal | Unreliable | Profile/3/4 view detected |
| Very uneven lighting | Unreliable | Strong shadows on face |
| Too dark/overexposed | Unreliable | Extreme brightness |
| Low resolution | Unreliable | Image < 200x200 |
| Slight angle | Warning | Minor pose deviation |
| Lighting imbalance | Warning | Some shadow variation |

## Output Example

```
HEADLINE
"Striking, warm-toned face with eye-forward composition"

PROPORTIONAL SIGNATURE
  - Wide-set eyes create an open, approachable quality
  - Balanced forehead-to-face ratio
  - Harmoniously proportioned lips

SYMMETRY CHARACTER
  - Natural asymmetry adds character and interest
  - Left-dominant pattern adds natural expressiveness

GEOMETRIC ARCHETYPE
  - Long-angular face shape
  - Elongated with defined angles - often described as striking

COLOR CHARACTER
  - Warm undertones - golden and peachy hues dominate
  - High contrast - dark features against lighter skin create natural drama
```

## Philosophy

Traditional facial analysis tools score faces against arbitrary beauty standards. FaceCraft takes a different approach:

- **No beauty scores** - Every face is unique, not better or worse
- **Descriptive, not prescriptive** - We describe what IS, not what should be
- **Celebrates asymmetry** - Perfectly symmetrical faces are rare and often less interesting
- **Honest about limitations** - When image quality affects accuracy, we say so

## License

MIT
