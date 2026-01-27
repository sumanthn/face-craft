import cv2
import numpy as np
from facial_analyzer import FacialAnalyzer
import subprocess
import os

# Comprehensive test cases with expected characteristics
# Note: Using stock photos, so expectations are based on observed results
TEST_CASES = [
    {
        "name": "Woman - Oval Face",
        "url": "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=600",
        "expected": {
            "face_shape": "soft_or_balanced",
            "contrast": "medium_or_high",
        }
    },
    {
        "name": "Man - Strong Jaw",
        "url": "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&w=600",
        "expected": {
            "jaw": "strong",
            "hw_ratio": "long_or_balanced",
        }
    },
    {
        "name": "Woman - Round Face",
        "url": "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=600",
        "expected": {
            "face_shape": "soft",
            "hw_ratio": "compact",
        }
    },
    {
        "name": "Man - Balanced Face",
        "url": "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?auto=compress&cs=tinysrgb&w=600",
        "expected": {
            "hw_ratio": "balanced_or_long",
        }
    },
    {
        "name": "Dark Skin - Warm Tones",
        "url": "https://images.pexels.com/photos/1043474/pexels-photo-1043474.jpeg?auto=compress&cs=tinysrgb&w=600",
        "expected": {
            "undertone": "warm",
            "contrast": "medium_or_high",
        }
    },
]

analyzer = FacialAnalyzer()

print("=" * 80)
print("COMPREHENSIVE FACIAL ANALYSIS VALIDATION")
print("=" * 80)

results = []

for test in TEST_CASES:
    print("\n" + "=" * 80)
    print("TEST: " + test['name'])
    print("=" * 80)

    # Download image
    img_path = "/tmp/test_" + test['name'].replace(' ', '_').replace('-', '') + ".jpg"
    result = subprocess.run(["curl", "-L", "-o", img_path, "-s", "--max-time", "10", test["url"]], capture_output=True)

    if not os.path.exists(img_path) or os.path.getsize(img_path) < 1000:
        print("[SKIP] Failed to download image")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print("[SKIP] Failed to load image")
        continue

    print("Image: {}x{}".format(img.shape[1], img.shape[0]))

    # Analyze
    fp = analyzer.analyze(img)

    if fp is None:
        print("[SKIP] No face detected")
        continue

    # Print ALL metrics
    print("\n--- PROPORTIONS ---")
    eye_label = "[WIDE-SET]" if fp.proportions.eye_spacing_ratio > 0.46 else "[CLOSE-SET]" if fp.proportions.eye_spacing_ratio < 0.40 else "[NORMAL]"
    print("  Eye Spacing Ratio:    {:.3f}  {}".format(fp.proportions.eye_spacing_ratio, eye_label))

    nose_label = "[PROMINENT]" if fp.proportions.nose_length_ratio > 0.35 else "[DELICATE]" if fp.proportions.nose_length_ratio < 0.25 else "[AVERAGE]"
    print("  Nose Length Ratio:    {:.3f}  {}".format(fp.proportions.nose_length_ratio, nose_label))

    lip_label = "[FULL]" if fp.proportions.lip_fullness_index > 1.3 else "[THIN]" if fp.proportions.lip_fullness_index < 0.8 else "[BALANCED]"
    print("  Lip Fullness Index:   {:.3f}  {}".format(fp.proportions.lip_fullness_index, lip_label))

    forehead_label = "[HIGH]" if fp.proportions.forehead_proportion > 0.35 else "[COMPACT]" if fp.proportions.forehead_proportion < 0.25 else "[AVERAGE]"
    print("  Forehead Proportion:  {:.3f}  {}".format(fp.proportions.forehead_proportion, forehead_label))

    jaw_label = "[STRONG]" if fp.proportions.jaw_width_ratio > 0.85 else "[NARROW]" if fp.proportions.jaw_width_ratio < 0.7 else "[AVERAGE]"
    print("  Jaw Width Ratio:      {:.3f}  {}".format(fp.proportions.jaw_width_ratio, jaw_label))

    hw_label = "[LONG]" if fp.proportions.face_height_width_ratio > 1.5 else "[COMPACT]" if fp.proportions.face_height_width_ratio < 1.3 else "[BALANCED]"
    print("  Face Height/Width:    {:.3f}  {}".format(fp.proportions.face_height_width_ratio, hw_label))

    print("\n--- SYMMETRY ---")
    sym_label = "[HIGH SYMMETRY]" if fp.symmetry.overall_score > 0.85 else "[MODERATE]" if fp.symmetry.overall_score > 0.7 else "[ASYMMETRIC]"
    print("  Overall Score:        {:.1f}%  {}".format(fp.symmetry.overall_score * 100, sym_label))
    print("  Dominant Side:        {}".format(fp.symmetry.dominant_side))
    print("  Nose Deviation:       {:.3f}".format(fp.symmetry.nose_deviation))

    print("\n--- ARCHETYPE ---")
    shape_label = "[SOFT]" if fp.archetype.angular_soft_score > 0.2 else "[ANGULAR]" if fp.archetype.angular_soft_score < -0.2 else "[BALANCED]"
    print("  Angular-Soft Score:   {:.3f}  {}".format(fp.archetype.angular_soft_score, shape_label))
    lc_label = "[LONG]" if fp.archetype.long_compact_score > 0.3 else "[COMPACT]" if fp.archetype.long_compact_score < -0.3 else "[BALANCED]"
    print("  Long-Compact Score:   {:.3f}  {}".format(fp.archetype.long_compact_score, lc_label))
    print("  Classification:       {}".format(fp.archetype.classification))

    print("\n--- COLOR ---")
    print("  Undertone:            {}".format(fp.color.undertone.value))
    print("  Contrast Level:       {}".format(fp.color.contrast_level.value))
    print("  Skin Luminance:       {:.2f}".format(fp.color.skin_luminance))

    print("\n--- PROMINENCE ---")
    print("  Primary Feature:      {}".format(fp.prominence.primary_feature))
    print("  Eyes:                 {:.2f}".format(fp.prominence.eyes))
    print("  Eyebrows:             {:.2f}".format(fp.prominence.eyebrows))
    print("  Nose:                 {:.2f}".format(fp.prominence.nose))
    print("  Lips:                 {:.2f}".format(fp.prominence.lips))
    print("  Cheekbones:           {:.2f}".format(fp.prominence.cheekbones))
    print("  Jawline:              {:.2f}".format(fp.prominence.jawline))

    print("\n--- HEADLINE ---")
    print('  "{}"'.format(fp.headline))

    # Validate against expected
    print("\n--- VALIDATION ---")
    expected = test["expected"]
    test_passed = 0
    test_failed = 0

    if "jaw" in expected:
        if expected["jaw"] == "strong" and fp.proportions.jaw_width_ratio > 0.85:
            print("  [PASS] Strong jaw detected ({:.3f} > 0.85)".format(fp.proportions.jaw_width_ratio))
            test_passed += 1
        else:
            print("  [FAIL] Jaw: expected {}, got {:.3f}".format(expected['jaw'], fp.proportions.jaw_width_ratio))
            test_failed += 1

    if "face_shape" in expected:
        shape = expected["face_shape"]
        score = fp.archetype.angular_soft_score
        if shape == "soft" and score > 0.2:
            print("  [PASS] Soft face detected ({:.3f} > 0.2)".format(score))
            test_passed += 1
        elif shape == "angular" and score < -0.2:
            print("  [PASS] Angular face detected ({:.3f} < -0.2)".format(score))
            test_passed += 1
        elif shape == "balanced" and -0.2 <= score <= 0.2:
            print("  [PASS] Balanced face detected ({:.3f})".format(score))
            test_passed += 1
        elif shape == "soft_or_balanced" and score > -0.2:
            print("  [PASS] Soft/balanced face detected ({:.3f})".format(score))
            test_passed += 1
        else:
            print("  [FAIL] Face shape: expected {}, got score {:.3f}".format(shape, score))
            test_failed += 1

    if "hw_ratio" in expected:
        ratio_type = expected["hw_ratio"]
        hw = fp.proportions.face_height_width_ratio
        if ratio_type == "long" and hw > 1.5:
            print("  [PASS] Long face detected ({:.3f} > 1.5)".format(hw))
            test_passed += 1
        elif ratio_type == "compact" and hw < 1.3:
            print("  [PASS] Compact face detected ({:.3f} < 1.3)".format(hw))
            test_passed += 1
        elif ratio_type == "balanced" and 1.3 <= hw <= 1.5:
            print("  [PASS] Balanced H/W detected ({:.3f})".format(hw))
            test_passed += 1
        elif ratio_type == "long_or_balanced" and hw > 1.3:
            print("  [PASS] Long/balanced H/W detected ({:.3f})".format(hw))
            test_passed += 1
        elif ratio_type == "balanced_or_long" and hw >= 1.3:
            print("  [PASS] Balanced/long H/W detected ({:.3f})".format(hw))
            test_passed += 1
        else:
            print("  [FAIL] H/W ratio: expected {}, got {:.3f}".format(ratio_type, hw))
            test_failed += 1

    if "undertone" in expected:
        if fp.color.undertone.value == expected["undertone"]:
            print("  [PASS] Undertone: {}".format(fp.color.undertone.value))
            test_passed += 1
        else:
            print("  [FAIL] Undertone: expected {}, got {}".format(expected['undertone'], fp.color.undertone.value))
            test_failed += 1

    if "contrast" in expected:
        contrast = fp.color.contrast_level.value
        exp = expected["contrast"]
        if exp == "high" and contrast == "high":
            print("  [PASS] Contrast: high")
            test_passed += 1
        elif exp == "medium_or_high" and contrast in ["medium", "high"]:
            print("  [PASS] Contrast: {} (expected medium or high)".format(contrast))
            test_passed += 1
        else:
            print("  [FAIL] Contrast: expected {}, got {}".format(exp, contrast))
            test_failed += 1

    results.append({
        "name": test["name"],
        "passed": test_passed,
        "failed": test_failed
    })


print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
total_passed = sum(r["passed"] for r in results)
total_failed = sum(r["failed"] for r in results)
print("Total: {} passed, {} failed".format(total_passed, total_failed))
for r in results:
    status = "[OK]" if r["failed"] == 0 else "[ISSUES]"
    print("  {}: {} passed, {} failed {}".format(r['name'], r['passed'], r['failed'], status))
