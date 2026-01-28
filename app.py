"""
Facial Uniqueness Analyzer - Gradio Web Interface
Upload a face image and discover what makes it uniquely distinctive.
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from facial_analyzer import FacialAnalyzer, FacialFingerprint


# Initialize analyzer
analyzer = FacialAnalyzer()


def format_report(fingerprint: FacialFingerprint) -> str:
    """Format the fingerprint as a beautiful text report."""

    # Check for quality warnings
    warning_section = ""
    if fingerprint.quality_warnings:
        warning_section = """
╔══════════════════════════════════════════════════════════════════╗
║                    ⚠️  IMAGE QUALITY NOTICE                       ║
╠══════════════════════════════════════════════════════════════════╣
"""
        for qw in fingerprint.quality_warnings:
            if qw.severity == "unreliable":
                warning_section += f"\n  [!] {qw.message}"
                warning_section += f"\n      Affected: {', '.join(qw.affected_metrics)}"
            else:
                warning_section += f"\n  [i] {qw.message}"

        if not fingerprint.is_reliable:
            warning_section += """

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Results below may be INACCURATE due to image quality issues.
  For reliable analysis, please use a well-lit, front-facing photo.
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        warning_section += "\n╚══════════════════════════════════════════════════════════════════╝\n"

    reliability_note = "" if fingerprint.is_reliable else " (⚠️ RESULTS MAY BE INACCURATE)"

    report = f"""{warning_section}
╔══════════════════════════════════════════════════════════════════╗
║                    YOUR FACIAL FINGERPRINT{reliability_note:^24}║
╠══════════════════════════════════════════════════════════════════╣

  HEADLINE
  "{fingerprint.headline}"

╠══════════════════════════════════════════════════════════════════╣

{fingerprint.detailed_report}

╠══════════════════════════════════════════════════════════════════╣

  RAW METRICS (for the curious)

  Proportions:
    Eye Spacing Ratio: {fingerprint.proportions.eye_spacing_ratio:.3f}
    Nose Length Ratio: {fingerprint.proportions.nose_length_ratio:.3f}
    Lip Fullness Index: {fingerprint.proportions.lip_fullness_index:.3f}
    Forehead Proportion: {fingerprint.proportions.forehead_proportion:.3f}
    Jaw Width Ratio: {fingerprint.proportions.jaw_width_ratio:.3f}
    Face Height/Width: {fingerprint.proportions.face_height_width_ratio:.3f}

  Symmetry:
    Overall Score: {fingerprint.symmetry.overall_score:.2%}
    Dominant Side: {fingerprint.symmetry.dominant_side}
    Nose Deviation: {fingerprint.symmetry.nose_deviation:.3f}

  Archetype:
    Angular-Soft Score: {fingerprint.archetype.angular_soft_score:.3f}
    Long-Compact Score: {fingerprint.archetype.long_compact_score:.3f}
    Classification: {fingerprint.archetype.classification}

  Color:
    Undertone: {fingerprint.color.undertone.value}
    Contrast: {fingerprint.color.contrast_level.value}
    Skin Luminance: {fingerprint.color.skin_luminance:.2f}

╚══════════════════════════════════════════════════════════════════╝
"""
    return report


def analyze_face(image):
    """Main analysis function for Gradio."""

    if image is None:
        return None, "Please upload an image."

    # Convert PIL Image to OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Analyze the face
    fingerprint = analyzer.analyze(image)

    if fingerprint is None:
        return None, "No face detected in the image. Please upload a clear photo with a visible face."

    # Generate visualization
    annotated = analyzer.visualize_landmarks(image)
    if annotated is not None:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Generate report
    report = format_report(fingerprint)

    return annotated, report


# Create Gradio interface
with gr.Blocks(title="Facial Uniqueness Analyzer") as demo:
    gr.Markdown("""
    # Facial Uniqueness Analyzer

    **Discover what makes your face distinctly yours.**

    This tool celebrates uniqueness rather than scoring beauty. Every face is a composition —
    we're not judging the art, we're describing what the artist did.

    Upload a clear, front-facing photo to get your facial fingerprint.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Your Photo",
                type="pil",
                height=400
            )
            analyze_btn = gr.Button("Analyze My Face", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detected Landmarks",
                height=400
            )

    output_report = gr.Textbox(
        label="Your Facial Fingerprint",
        lines=40,
        max_lines=50,
        elem_classes=["report-box"]
    )

    analyze_btn.click(
        fn=analyze_face,
        inputs=[input_image],
        outputs=[output_image, output_report]
    )

    gr.Markdown("""
    ---

    ### What We Analyze

    - **Proportional Signature** — The unique geometry of your features
    - **Symmetry Character** — Your distinctive asymmetry patterns (not deficiencies!)
    - **Feature Prominence** — What draws the eye first
    - **Geometric Archetype** — Your face shape on the angular-soft and long-compact spectrums
    - **Color Character** — Undertones, contrast, and luminance

    ### Privacy Note

    Your images are processed in real-time and are not stored. Analysis happens entirely on-device.

    ---

    *Built with InsightFace (GPU) / MediaPipe (fallback) • No beauty scores, just uniqueness*
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8090,
        share=True
    )
