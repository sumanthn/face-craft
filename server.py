"""
FaceCraft - FastAPI Backend Server
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from pathlib import Path
import httpx
import json

from facial_analyzer import FacialAnalyzer

app = FastAPI(title="FaceCraft", description="Facial Uniqueness Analyzer")

# DeepSeek API for natural summaries
DEEPSEEK_API_KEY = "sk-37c4a570637143c6b34b01aadbc0e29e"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
print("Initializing FaceCraft analyzer...")
analyzer = FacialAnalyzer()
print("Analyzer ready!")

# Serve static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")


def get_unique_features(fingerprint) -> list:
    """Extract what makes this face unique - always returns meaningful features."""
    unique = []

    # Primary feature - ALWAYS include this first
    primary = fingerprint.prominence.primary_feature
    feature_details = {
        "eyes": ("Expressive Eyes", "Your eyes are the natural focal point, drawing attention first"),
        "eyebrows": ("Defined Brows", "Strong, well-defined brows frame your face beautifully"),
        "nose": ("Distinctive Nose", "Your nose adds character and anchors your profile"),
        "lips": ("Expressive Lips", "Your lips are a natural focal point with great expression"),
        "cheekbones": ("Sculpted Cheekbones", "High cheekbones create elegant facial structure"),
        "jawline": ("Strong Jawline", "A defined jawline anchors your face with strength")
    }
    name, desc = feature_details.get(primary, (primary.title(), f"Your {primary} stand out"))
    unique.append({"feature": name, "description": desc})

    # Eye spacing
    eye_ratio = fingerprint.proportions.eye_spacing_ratio
    if eye_ratio > 0.47:
        unique.append({"feature": "Wide-Set Eyes", "description": "Eyes set wider apart, creating an open, dreamy look"})
    elif eye_ratio < 0.39:
        unique.append({"feature": "Close-Set Eyes", "description": "Eyes closer together, adding intensity and focus"})

    # Face shape - angular vs soft
    angular = fingerprint.archetype.angular_soft_score
    if angular < -0.15:
        unique.append({"feature": "Angular Features", "description": "Strong bone structure with defined angles"})
    elif angular > 0.15:
        unique.append({"feature": "Soft Features", "description": "Gentle, rounded contours that appear warm and approachable"})

    # Face length
    long_compact = fingerprint.archetype.long_compact_score
    if long_compact > 0.25:
        unique.append({"feature": "Elongated Face", "description": "Elegant, longer face proportions"})
    elif long_compact < -0.25:
        unique.append({"feature": "Compact Face", "description": "Well-proportioned, balanced face shape"})

    # Symmetry (only mention if notable)
    symmetry = fingerprint.symmetry.overall_score
    if symmetry > 0.96:
        unique.append({"feature": "Exceptional Symmetry", "description": "Remarkably balanced, harmonious features"})
    elif symmetry < 0.88:
        unique.append({"feature": "Distinctive Character", "description": "Natural asymmetry that adds unique personality"})

    # Contrast
    contrast = fingerprint.color.contrast_level.value
    if contrast == "high":
        unique.append({"feature": "High Contrast", "description": "Striking contrast between features and skin tone"})
    elif contrast == "low":
        unique.append({"feature": "Soft Harmony", "description": "Gentle, cohesive tonal range across features"})

    # Undertone
    undertone = fingerprint.color.undertone.value
    if undertone == "warm":
        unique.append({"feature": "Warm Undertones", "description": "Golden, sun-kissed skin quality"})
    elif undertone == "cool":
        unique.append({"feature": "Cool Undertones", "description": "Elegant pink or blue-based skin tones"})

    return unique


def get_simple_scores(fingerprint) -> dict:
    """Get clean, simple scores (no long decimals)."""
    return {
        "symmetry": int(round(fingerprint.symmetry.overall_score * 100)),
        "eye_spacing": int(round(fingerprint.proportions.eye_spacing_ratio * 100)),
        "face_shape": fingerprint.archetype.classification,
        "contrast": fingerprint.color.contrast_level.value.title(),
        "undertone": fingerprint.color.undertone.value.title(),
        "primary_feature": fingerprint.prominence.primary_feature.replace("_", " ").title(),
    }


def get_feature_shapes(fingerprint) -> dict:
    """Get detailed feature shape analysis from our model."""
    shapes = fingerprint.shapes
    return {
        "eyes": {
            "shape": shapes.eye_shape.title(),
            "size": shapes.eye_size.title(),
            "description": get_eye_description(shapes.eye_shape, shapes.eye_size)
        },
        "brows": {
            "shape": shapes.brow_shape.title(),
            "thickness": shapes.brow_thickness.title(),
            "description": get_brow_description(shapes.brow_shape, shapes.brow_thickness)
        },
        "nose": {
            "shape": shapes.nose_shape.title(),
            "tip": shapes.nose_tip.title(),
            "description": get_nose_description(shapes.nose_shape, shapes.nose_tip)
        },
        "lips": {
            "shape": shapes.lip_shape.title(),
            "ratio": shapes.lip_ratio.title(),
            "description": get_lip_description(shapes.lip_shape, shapes.lip_ratio)
        },
        "chin": {
            "shape": shapes.chin_shape.title(),
            "description": get_chin_description(shapes.chin_shape)
        },
        "cheekbones": {
            "prominence": shapes.cheekbone_prominence.title(),
            "description": get_cheekbone_description(shapes.cheekbone_prominence)
        }
    }


def get_eye_description(shape, size):
    descriptions = {
        ("round", "large"): "Striking, expressive round eyes that draw immediate attention",
        ("round", "medium"): "Soft, round eyes with a warm, inviting quality",
        ("round", "small"): "Delicate round eyes with an alert, curious expression",
        ("almond", "large"): "Elegant, elongated almond eyes - classically beautiful",
        ("almond", "medium"): "Well-proportioned almond eyes with timeless appeal",
        ("almond", "small"): "Refined almond eyes with subtle sophistication",
        ("hooded", "large"): "Mysterious hooded eyes with captivating depth",
        ("hooded", "medium"): "Sultry hooded eyes that create an alluring look",
        ("hooded", "small"): "Intense hooded eyes with focused appeal",
        ("upturned", "large"): "Bright, upturned eyes with a naturally cheerful look",
        ("upturned", "medium"): "Cat-like upturned eyes with elegant lift",
        ("upturned", "small"): "Playful upturned eyes with pixie-like charm",
        ("downturned", "large"): "Soulful downturned eyes with emotional depth",
        ("downturned", "medium"): "Gentle downturned eyes with a thoughtful expression",
        ("downturned", "small"): "Soft downturned eyes with quiet intensity",
    }
    return descriptions.get((shape, size), f"{size.title()} {shape} eyes")


def get_brow_description(shape, thickness):
    descriptions = {
        ("arched", "thick"): "Bold, dramatically arched brows that frame the face powerfully",
        ("arched", "medium"): "Elegantly arched brows with refined definition",
        ("arched", "thin"): "Delicately arched brows with graceful lines",
        ("curved", "thick"): "Strong, naturally curved brows with full body",
        ("curved", "medium"): "Soft, gently curved brows with balanced shape",
        ("curved", "thin"): "Fine, curved brows with understated elegance",
        ("straight", "thick"): "Bold, straight brows creating a strong horizontal line",
        ("straight", "medium"): "Clean, straight brows with modern appeal",
        ("straight", "thin"): "Sleek, straight brows with minimalist beauty",
        ("s-shaped", "thick"): "Distinctive S-shaped brows with dramatic character",
        ("s-shaped", "medium"): "Unique S-curved brows with artistic flair",
        ("s-shaped", "thin"): "Subtle S-shaped brows with delicate definition",
    }
    return descriptions.get((shape, thickness), f"{thickness.title()} {shape} brows")


def get_nose_description(shape, tip):
    descriptions = {
        ("straight", "rounded"): "Classic straight nose with a soft, rounded tip",
        ("straight", "pointed"): "Refined straight nose with elegant definition",
        ("straight", "upturned"): "Straight nose with a charming upturned tip",
        ("straight", "downturned"): "Distinguished straight nose with strong profile",
        ("narrow", "rounded"): "Delicate narrow nose with soft contours",
        ("narrow", "pointed"): "Elegant narrow nose with refined tip",
        ("narrow", "upturned"): "Petite nose with an endearing upturned tip",
        ("narrow", "downturned"): "Slim nose with graceful downward curve",
        ("wide", "rounded"): "Strong nose with soft, rounded features",
        ("wide", "pointed"): "Bold nose with defined tip",
        ("wide", "upturned"): "Characterful nose with upturned tip",
        ("wide", "downturned"): "Prominent nose with distinguished profile",
        ("aquiline", "rounded"): "Noble aquiline nose with soft tip",
        ("aquiline", "pointed"): "Striking aquiline profile with sharp definition",
        ("aquiline", "downturned"): "Regal aquiline nose with dramatic curve",
    }
    return descriptions.get((shape, tip), f"{shape.title()} nose with {tip} tip")


def get_lip_description(shape, ratio):
    descriptions = {
        ("full", "balanced"): "Beautifully full lips with perfect proportions",
        ("full", "top-heavy"): "Lush lips with a prominent upper lip",
        ("full", "bottom-heavy"): "Sensual lips with a fuller lower lip",
        ("thin", "balanced"): "Refined, delicate lips with elegant lines",
        ("thin", "top-heavy"): "Subtle lips with defined upper contour",
        ("thin", "bottom-heavy"): "Graceful lips with soft lower fullness",
        ("heart", "balanced"): "Romantic heart-shaped lips with lovely symmetry",
        ("heart", "top-heavy"): "Cupid's bow lips with pronounced upper shape",
        ("heart", "bottom-heavy"): "Sweet heart-shaped lips with pouty lower lip",
        ("bow-shaped", "balanced"): "Classic bow-shaped lips with perfect definition",
        ("bow-shaped", "top-heavy"): "Dramatic cupid's bow with elegant peaks",
        ("wide", "balanced"): "Generous, expressive lips that enhance your smile",
    }
    return descriptions.get((shape, ratio), f"{shape.title()} lips")


def get_chin_description(shape):
    descriptions = {
        "pointed": "Defined pointed chin adding elegant length to the face",
        "rounded": "Soft rounded chin creating gentle, approachable appeal",
        "square": "Strong square chin with confident, defined structure",
    }
    return descriptions.get(shape, f"{shape.title()} chin")


def get_cheekbone_description(prominence):
    descriptions = {
        "high": "Prominent high cheekbones creating striking facial architecture",
        "medium": "Well-defined cheekbones with balanced structure",
        "low": "Soft cheekbone structure with gentle contours",
    }
    return descriptions.get(prominence, f"{prominence.title()} cheekbones")


def generate_summary(fingerprint) -> str:
    """Generate a natural summary directly from our model's analysis - no AI needed!"""
    shapes = fingerprint.shapes
    arch = fingerprint.archetype
    color = fingerprint.color
    sym = fingerprint.symmetry

    # Build specific, data-driven sentences
    parts = []

    # Lead with the most distinctive eye feature
    eye_desc = {
        ("round", "large"): "Your large, round eyes are immediately captivating",
        ("round", "medium"): "Your round eyes have a warm, expressive quality",
        ("round", "small"): "Your round eyes carry an alert, focused intensity",
        ("almond", "large"): "Your large almond eyes are classically striking",
        ("almond", "medium"): "Your almond-shaped eyes have elegant proportions",
        ("almond", "small"): "Your refined almond eyes convey quiet sophistication",
        ("hooded", "large"): "Your hooded eyes create a mysterious, magnetic look",
        ("hooded", "medium"): "Your hooded eyes have an alluring, sultry quality",
        ("hooded", "small"): "Your hooded eyes project focused intensity",
        ("upturned", "large"): "Your upturned eyes give you a bright, optimistic look",
        ("upturned", "medium"): "Your cat-like upturned eyes are distinctively elegant",
        ("upturned", "small"): "Your upturned eyes have a playful, pixie-like charm",
    }.get((shapes.eye_shape, shapes.eye_size), f"Your {shapes.eye_shape} eyes are distinctive")

    # Add brow context
    brow_adds = {
        ("s-shaped", "thick"): ", framed by bold, dramatically shaped brows",
        ("arched", "thick"): ", enhanced by strong arched brows",
        ("curved", "thick"): ", complemented by full, expressive brows",
        ("s-shaped", "thin"): ", with elegantly sculpted brows",
        ("arched", "thin"): ", accented by delicately arched brows",
    }.get((shapes.brow_shape, shapes.brow_thickness), "")

    parts.append(eye_desc + brow_adds + ".")

    # Second sentence: face structure
    structure_parts = []

    if shapes.cheekbone_prominence == "high":
        structure_parts.append("prominent cheekbones")

    if shapes.chin_shape == "pointed":
        structure_parts.append("a defined chin")
    elif shapes.chin_shape == "square":
        structure_parts.append("a strong jawline")

    if "angular" in arch.classification.lower():
        structure_parts.append("angular bone structure")
    elif "soft" in arch.classification.lower():
        structure_parts.append("soft, approachable contours")

    if structure_parts:
        parts.append(f"Your face features {', '.join(structure_parts[:2])}.")

    # Third sentence: distinctive details
    details = []

    nose_detail = {
        ("narrow", "pointed"): "an elegantly refined nose",
        ("narrow", "upturned"): "a charming upturned nose",
        ("wide", "rounded"): "a strong, characterful nose",
        ("straight", "pointed"): "a classically straight nose",
    }.get((shapes.nose_shape, shapes.nose_tip))
    if nose_detail:
        details.append(nose_detail)

    lip_detail = {
        ("full", "balanced"): "beautifully full lips",
        ("full", "top-heavy"): "full lips with a defined upper lip",
        ("full", "bottom-heavy"): "sensual, pouty lips",
        ("heart", "balanced"): "romantic heart-shaped lips",
        ("bow-shaped", "balanced"): "a perfect cupid's bow",
    }.get((shapes.lip_shape, shapes.lip_ratio))
    if lip_detail:
        details.append(lip_detail)

    if details:
        parts.append(f"You have {' and '.join(details)}.")

    # Add color/contrast note if striking
    if color.contrast_level.value == "high":
        parts.append("Your high contrast coloring is naturally photogenic.")

    return " ".join(parts)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """Analyze an uploaded face image."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not read image file"}
            )

        # Analyze
        fingerprint = analyzer.analyze(image)

        if fingerprint is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No face detected. Please use a clear, front-facing photo."}
            )

        # Get all analysis from our model
        unique_features = get_unique_features(fingerprint)
        simple_scores = get_simple_scores(fingerprint)
        feature_shapes = get_feature_shapes(fingerprint)

        # Generate summary from our model - no external AI needed!
        summary = generate_summary(fingerprint)

        # Build response with rich feature data
        result = {
            "summary": summary,
            "feature_shapes": feature_shapes,
            "unique_features": unique_features,
            "scores": simple_scores,
            "is_reliable": fingerprint.is_reliable,
            "quality_warnings": [
                {
                    "message": qw.message,
                    "severity": qw.severity,
                }
                for qw in fingerprint.quality_warnings
            ] if fingerprint.quality_warnings else []
        }

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "FaceCraft"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
