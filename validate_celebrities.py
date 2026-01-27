"""
Validate facial analyzer against known celebrity characteristics.
"""
import cv2
import os
from facial_analyzer import FacialAnalyzer

# Celebrity validation matrix with known characteristics
CELEBRITIES = {
    # Wide-set eyes
    "anya_taylor_joy": {
        "file": "anya_taylor_joy.jpg",
        "expected_eye_spacing": "wide",  # Famous for wide-set eyes
        "expected_face_shape": "angular_or_balanced",
    },
    "amanda_seyfried": {
        "file": "amanda_seyfried.jpg",
        "expected_eye_spacing": "wide",  # Known for large, wide-set eyes
    },

    # Close-set eyes
    "mila_kunis": {
        "file": "mila_kunis.jpg",
        "expected_eye_spacing": "close_or_normal",  # Closer-set eyes
    },

    # Angular faces
    "tilda_swinton": {
        "file": "tilda_swinton.jpg",
        "expected_face_shape": "angular",  # Known for angular, striking features
    },
    "cate_blanchett": {
        "file": "cate_blanchett.jpg",
        "expected_face_shape": "angular_or_balanced",  # Defined cheekbones
    },
    "benedict_cumberbatch": {
        "file": "benedict_cumberbatch.jpg",
        "expected_face_shape": "angular_or_balanced",  # Angular, distinctive
    },
    "adam_driver": {
        "file": "adam_driver.jpg",
        "expected_face_shape": "angular",  # Very angular, long face
        "expected_hw_ratio": "long",
    },

    # Soft/round faces
    "emma_stone": {
        "file": "emma_stone.jpg",
        "expected_face_shape": "soft_or_balanced",  # Rounder, softer features
    },
    "margot_robbie": {
        "file": "margot_robbie.jpg",
        "expected_face_shape": "soft_or_balanced",  # Heart-shaped, softer
    },
    "zendaya": {
        "file": "zendaya.jpg",
        "expected_face_shape": "balanced",  # Balanced oval
    },

    # Long faces
    "robert_pattinson": {
        "file": "robert_pattinson.jpg",
        "expected_hw_ratio": "long_or_balanced",
        "expected_contrast": "high",  # Very pale with dark hair
    },

    # High contrast
    "lupita_nyongo": {
        "file": "lupita_nyongo.jpg",
        "expected_contrast": "high",  # Deep skin, very dark features
        "expected_undertone": "warm",
    },

    # Indian celebrities
    "aishwarya_rai": {
        "file": "aishwarya_rai.jpg",
        "expected_eye_spacing": "normal_or_wide",  # Famous for eyes
        "expected_contrast": "medium_or_high",
    },
    "shah_rukh_khan": {
        "file": "shah_rukh_khan.jpg",
        "expected_contrast": "medium_or_high",
    },

    # Other notable faces
    "leonardo_dicaprio": {
        "file": "leonardo_dicaprio.jpg",
        "expected_face_shape": "balanced",
    },
    "ryan_gosling": {
        "file": "ryan_gosling.jpg",
        "expected_face_shape": "balanced",
    },
    "natalie_portman": {
        "file": "natalie_portman.jpg",
        "expected_face_shape": "soft_or_balanced",
    },
    "idris_elba": {
        "file": "idris_elba.jpg",
        "expected_face_shape": "angular_or_balanced",  # Strong jaw
    },
    "denzel_washington": {
        "file": "denzel_washington.jpg",
        "expected_contrast": "medium_or_high",
    },
}

def validate_eye_spacing(ratio, expected):
    """Check eye spacing matches expectation."""
    if expected == "wide":
        return ratio > 0.46, "wide" if ratio > 0.46 else ("normal" if ratio >= 0.40 else "close")
    elif expected == "close":
        return ratio < 0.40, "close" if ratio < 0.40 else ("normal" if ratio <= 0.46 else "wide")
    elif expected == "close_or_normal":
        return ratio <= 0.46, "close" if ratio < 0.40 else ("normal" if ratio <= 0.46 else "wide")
    elif expected == "normal_or_wide":
        return ratio >= 0.40, "close" if ratio < 0.40 else ("normal" if ratio <= 0.46 else "wide")
    elif expected == "normal":
        return 0.40 <= ratio <= 0.46, "close" if ratio < 0.40 else ("normal" if ratio <= 0.46 else "wide")
    return True, "unknown"

def validate_face_shape(score, expected):
    """Check face shape (angular/soft) matches expectation."""
    actual = "angular" if score < -0.2 else ("soft" if score > 0.2 else "balanced")
    if expected == "angular":
        return score < -0.2, actual
    elif expected == "soft":
        return score > 0.2, actual
    elif expected == "balanced":
        return -0.2 <= score <= 0.2, actual
    elif expected == "angular_or_balanced":
        return score <= 0.2, actual
    elif expected == "soft_or_balanced":
        return score >= -0.2, actual
    return True, actual

def validate_hw_ratio(ratio, expected):
    """Check height/width ratio matches expectation."""
    actual = "long" if ratio > 1.5 else ("compact" if ratio < 1.3 else "balanced")
    if expected == "long":
        return ratio > 1.5, actual
    elif expected == "compact":
        return ratio < 1.3, actual
    elif expected == "balanced":
        return 1.3 <= ratio <= 1.5, actual
    elif expected == "long_or_balanced":
        return ratio >= 1.3, actual
    return True, actual

def validate_contrast(level, expected):
    """Check contrast level matches expectation."""
    if expected == "high":
        return level == "high", level
    elif expected == "medium_or_high":
        return level in ["medium", "high"], level
    elif expected == "low":
        return level == "low", level
    return True, level

def validate_undertone(tone, expected):
    """Check undertone matches expectation."""
    return tone == expected, tone

def main():
    print("=" * 80)
    print("CELEBRITY VALIDATION MATRIX")
    print("=" * 80)

    img_dir = "/workspace/facial-uniqueness/test-images/celebrities"

    # Check which images exist
    available = []
    missing = []
    for name, data in CELEBRITIES.items():
        path = os.path.join(img_dir, data["file"])
        if os.path.exists(path):
            available.append(name)
        else:
            missing.append(name)

    print("\nAvailable: {} celebrities".format(len(available)))
    if missing:
        print("Missing: {}".format(", ".join(missing)))

    print("\nInitializing analyzer...")
    analyzer = FacialAnalyzer()

    results = []

    for name in sorted(available):
        data = CELEBRITIES[name]
        path = os.path.join(img_dir, data["file"])

        print("\n" + "-" * 60)
        display_name = name.replace("_", " ").title()
        print("Testing: {}".format(display_name))
        print("-" * 60)

        img = cv2.imread(path)
        if img is None:
            print("  [SKIP] Could not load image")
            continue

        fp = analyzer.analyze(img)
        if fp is None:
            print("  [SKIP] No face detected")
            continue

        passed = 0
        failed = 0

        # Print metrics
        print("  Eye Spacing:     {:.3f}".format(fp.proportions.eye_spacing_ratio), end="")
        if fp.proportions.eye_spacing_ratio > 0.46:
            print(" [WIDE]")
        elif fp.proportions.eye_spacing_ratio < 0.40:
            print(" [CLOSE]")
        else:
            print(" [NORMAL]")

        print("  Angular/Soft:    {:.3f}".format(fp.archetype.angular_soft_score), end="")
        if fp.archetype.angular_soft_score < -0.2:
            print(" [ANGULAR]")
        elif fp.archetype.angular_soft_score > 0.2:
            print(" [SOFT]")
        else:
            print(" [BALANCED]")

        print("  H/W Ratio:       {:.3f}".format(fp.proportions.face_height_width_ratio), end="")
        if fp.proportions.face_height_width_ratio > 1.5:
            print(" [LONG]")
        elif fp.proportions.face_height_width_ratio < 1.3:
            print(" [COMPACT]")
        else:
            print(" [BALANCED]")

        print("  Contrast:        {}".format(fp.color.contrast_level.value))
        print("  Undertone:       {}".format(fp.color.undertone.value))

        # Validate
        print("  --- Validation ---")

        if "expected_eye_spacing" in data:
            ok, actual = validate_eye_spacing(fp.proportions.eye_spacing_ratio, data["expected_eye_spacing"])
            if ok:
                print("  [PASS] Eye spacing: {} (expected {})".format(actual, data["expected_eye_spacing"]))
                passed += 1
            else:
                print("  [FAIL] Eye spacing: {} (expected {})".format(actual, data["expected_eye_spacing"]))
                failed += 1

        if "expected_face_shape" in data:
            ok, actual = validate_face_shape(fp.archetype.angular_soft_score, data["expected_face_shape"])
            if ok:
                print("  [PASS] Face shape: {} (expected {})".format(actual, data["expected_face_shape"]))
                passed += 1
            else:
                print("  [FAIL] Face shape: {} (expected {})".format(actual, data["expected_face_shape"]))
                failed += 1

        if "expected_hw_ratio" in data:
            ok, actual = validate_hw_ratio(fp.proportions.face_height_width_ratio, data["expected_hw_ratio"])
            if ok:
                print("  [PASS] H/W ratio: {} (expected {})".format(actual, data["expected_hw_ratio"]))
                passed += 1
            else:
                print("  [FAIL] H/W ratio: {} (expected {})".format(actual, data["expected_hw_ratio"]))
                failed += 1

        if "expected_contrast" in data:
            ok, actual = validate_contrast(fp.color.contrast_level.value, data["expected_contrast"])
            if ok:
                print("  [PASS] Contrast: {} (expected {})".format(actual, data["expected_contrast"]))
                passed += 1
            else:
                print("  [FAIL] Contrast: {} (expected {})".format(actual, data["expected_contrast"]))
                failed += 1

        if "expected_undertone" in data:
            ok, actual = validate_undertone(fp.color.undertone.value, data["expected_undertone"])
            if ok:
                print("  [PASS] Undertone: {} (expected {})".format(actual, data["expected_undertone"]))
                passed += 1
            else:
                print("  [FAIL] Undertone: {} (expected {})".format(actual, data["expected_undertone"]))
                failed += 1

        results.append({
            "name": display_name,
            "passed": passed,
            "failed": failed,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total = total_passed + total_failed

    print("\nTotal: {}/{} tests passed ({:.1f}%)".format(total_passed, total, 100*total_passed/total if total > 0 else 0))
    print("\nBy Celebrity:")

    for r in sorted(results, key=lambda x: x["failed"], reverse=True):
        status = "PASS" if r["failed"] == 0 else "FAIL"
        tests = r["passed"] + r["failed"]
        print("  [{:4s}] {:25s} {}/{}".format(status, r["name"], r["passed"], tests))


if __name__ == "__main__":
    main()
