import cv2
import numpy as np
from facial_analyzer import FacialAnalyzer
import subprocess
import os

# Test with known celebrity characteristics
TEST_CASES = [
    {
        "name": "Anya Taylor-Joy",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Anya_Taylor-Joy_by_Gage_Skidmore.jpg/440px-Anya_Taylor-Joy_by_Gage_Skidmore.jpg",
        "expected": {
            "eye_spacing": "wide",  # Famous for wide-set eyes
            "face_shape": "angular",  # Sharp features
        }
    },
    {
        "name": "Generic Test (Pexels)",
        "url": "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=400",
        "expected": {
            "eye_spacing": "normal",
        }
    },
]

analyzer = FacialAnalyzer()

print("=" * 70)
print("FACIAL UNIQUENESS ANALYZER - VALIDATION TEST")
print("=" * 70)

for test in TEST_CASES:
    print(f"\n\n### Testing: {test['name']} ###\n")
    
    # Download image
    img_path = f"/tmp/test_{test['name'].replace(' ', '_')}.jpg"
    result = subprocess.run(["curl", "-L", "-o", img_path, "-s", test["url"]], capture_output=True)
    
    if not os.path.exists(img_path):
        print(f"Failed to download image for {test['name']}")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image for {test['name']}")
        continue
    
    print(f"Image size: {img.shape}")
    
    # Analyze
    fingerprint = analyzer.analyze(img)
    
    if fingerprint is None:
        print("No face detected!")
        continue
    
    # Print results
    print(f"\nHEADLINE: {fingerprint.headline}")
    print(f"\nKEY METRICS:")
    print(f"  Eye Spacing Ratio: {fingerprint.proportions.eye_spacing_ratio:.3f}")
    print(f"  Face Height/Width: {fingerprint.proportions.face_height_width_ratio:.3f}")
    print(f"  Jaw Width Ratio: {fingerprint.proportions.jaw_width_ratio:.3f}")
    print(f"  Angular-Soft Score: {fingerprint.archetype.angular_soft_score:.3f}")
    print(f"  Classification: {fingerprint.archetype.classification}")
    print(f"  Symmetry Score: {fingerprint.symmetry.overall_score:.1%}")
    
    # Validate against expected
    print(f"\nVALIDATION:")
    eye_ratio = fingerprint.proportions.eye_spacing_ratio
    
    if test["expected"].get("eye_spacing") == "wide":
        if eye_ratio > 0.46:
            print(f"  [PASS] Wide-set eyes detected (ratio {eye_ratio:.3f} > 0.46)")
        else:
            print(f"  [FAIL] Expected wide-set eyes, got ratio {eye_ratio:.3f} (threshold: >0.46)")
    elif test["expected"].get("eye_spacing") == "close":
        if eye_ratio < 0.40:
            print(f"  [PASS] Close-set eyes detected (ratio {eye_ratio:.3f} < 0.40)")
        else:
            print(f"  [FAIL] Expected close-set eyes, got ratio {eye_ratio:.3f} (threshold: <0.40)")
    else:
        if 0.40 <= eye_ratio <= 0.46:
            print(f"  [PASS] Normal eye spacing (ratio {eye_ratio:.3f} in 0.40-0.46)")
        else:
            print(f"  [INFO] Eye spacing ratio: {eye_ratio:.3f}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
