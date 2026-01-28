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


async def generate_ai_summary(fingerprint, unique_features) -> str:
    """Use DeepSeek to generate a natural, personalized summary."""

    # Build context for the AI
    features_text = "\n".join([f"- {f['feature']}: {f['description']}" for f in unique_features])

    prompt = f"""You are a friendly facial feature analyst. Based on these detected features, write a SHORT, engaging 2-3 sentence summary about what makes this person's face unique and beautiful. Be positive and specific. Don't use generic phrases.

Detected unique features:
{features_text}

Face classification: {fingerprint.archetype.classification}
Symmetry score: {int(fingerprint.symmetry.overall_score * 100)}%
Primary feature: {fingerprint.prominence.primary_feature}

Write a warm, personalized summary (2-3 sentences only):"""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"DeepSeek API error: {e}")

    # Fallback to basic summary if API fails
    return fingerprint.headline


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

        # Get unique features and simple scores
        unique_features = get_unique_features(fingerprint)
        simple_scores = get_simple_scores(fingerprint)

        # Generate AI summary
        ai_summary = await generate_ai_summary(fingerprint, unique_features)

        # Build response
        result = {
            "ai_summary": ai_summary,
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
