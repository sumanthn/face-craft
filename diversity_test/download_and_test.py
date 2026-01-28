"""
Download celebrity images and run facial analysis for bias testing.
Uses DuckDuckGo image search (no API key needed).
"""

import os
import sys
import json
import time
import requests
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from celebrities_list import CELEBRITIES, get_all_celebrities

# Directories
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# API endpoint
API_URL = "http://69.30.85.91:22088/analyze"

def search_image_url(celebrity_name: str) -> str:
    """Search for a celebrity face image using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Search for face/portrait images
            query = f"{celebrity_name} face portrait photo"
            results = list(ddgs.images(query, max_results=5))

            for result in results:
                url = result.get('image')
                if url and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    return url
    except Exception as e:
        print(f"  Search error for {celebrity_name}: {e}")
    return None


def download_image(celebrity_name: str, region: str) -> str:
    """Download a celebrity image, return path or None."""
    safe_name = celebrity_name.lower().replace(" ", "_").replace("'", "")
    image_path = IMAGES_DIR / region / f"{safe_name}.jpg"

    # Skip if already exists
    if image_path.exists() and image_path.stat().st_size > 1000:
        return str(image_path)

    # Create region directory
    (IMAGES_DIR / region).mkdir(exist_ok=True)

    # Search for image
    url = search_image_url(celebrity_name)
    if not url:
        print(f"  No image found for {celebrity_name}")
        return None

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200 and len(response.content) > 5000:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return str(image_path)
    except Exception as e:
        print(f"  Download error for {celebrity_name}: {e}")

    return None


def analyze_image(image_path: str, celebrity_name: str, region: str) -> dict:
    """Send image to API for analysis."""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(API_URL, files=files, timeout=60)

            if response.status_code == 200:
                result = response.json()
                result['celebrity'] = celebrity_name
                result['region'] = region
                result['image_path'] = image_path
                return result
            else:
                return {
                    'celebrity': celebrity_name,
                    'region': region,
                    'error': f"API error: {response.status_code}"
                }
    except Exception as e:
        return {
            'celebrity': celebrity_name,
            'region': region,
            'error': str(e)
        }


def process_celebrity(celeb_info: dict) -> dict:
    """Download and analyze a single celebrity."""
    name = celeb_info['name']
    region = celeb_info['region']

    print(f"Processing: {name} ({region})")

    # Download
    image_path = download_image(name, region)
    if not image_path:
        return {'celebrity': name, 'region': region, 'error': 'No image downloaded'}

    # Analyze
    result = analyze_image(image_path, name, region)

    # Small delay to be nice to servers
    time.sleep(0.5)

    return result


def run_batch_test(max_workers: int = 5, limit: int = None):
    """Run the full diversity test."""
    all_celebs = get_all_celebrities()

    if limit:
        all_celebs = all_celebs[:limit]

    print(f"\n{'='*60}")
    print(f"FACECRAFT DIVERSITY TEST - {len(all_celebs)} celebrities")
    print(f"{'='*60}\n")

    results = []
    errors = []

    # Process sequentially to avoid rate limits
    for i, celeb in enumerate(all_celebs):
        print(f"\n[{i+1}/{len(all_celebs)}] ", end="")
        result = process_celebrity(celeb)

        if 'error' in result:
            errors.append(result)
        else:
            results.append(result)

        # Save progress every 20
        if (i + 1) % 20 == 0:
            save_results(results, errors)
            print(f"\n  >> Saved progress: {len(results)} successful, {len(errors)} errors\n")

    # Final save
    save_results(results, errors)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} analyzed, {len(errors)} errors")
    print(f"{'='*60}\n")

    return results, errors


def save_results(results: list, errors: list):
    """Save current results to JSON."""
    with open(RESULTS_DIR / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "errors.json", 'w') as f:
        json.dump(errors, f, indent=2)


def generate_report():
    """Generate bias analysis report from results."""
    results_file = RESULTS_DIR / "analysis_results.json"
    if not results_file.exists():
        print("No results file found. Run the test first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    print(f"\n{'='*70}")
    print("FACECRAFT DIVERSITY & BIAS ANALYSIS REPORT")
    print(f"{'='*70}\n")
    print(f"Total celebrities analyzed: {len(results)}\n")

    # Group by region
    by_region = {}
    for r in results:
        region = r.get('region', 'unknown')
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(r)

    # SYMMETRY ANALYSIS
    print("-" * 50)
    print("SYMMETRY SCORES BY REGION")
    print("-" * 50)

    region_symmetry = {}
    for region, celebs in sorted(by_region.items()):
        scores = [c['scores']['symmetry'] for c in celebs if 'scores' in c]
        if scores:
            avg = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            region_symmetry[region] = {'avg': avg, 'min': min_s, 'max': max_s, 'count': len(scores)}
            print(f"{region:25} | Avg: {avg:5.1f}% | Range: {min_s}-{max_s}% | n={len(scores)}")

    # Check for bias
    all_avgs = [v['avg'] for v in region_symmetry.values()]
    if all_avgs:
        global_avg = sum(all_avgs) / len(all_avgs)
        print(f"\nGlobal Average: {global_avg:.1f}%")

        # Flag if any region is >3% below average
        print("\nBias Check:")
        for region, data in region_symmetry.items():
            diff = data['avg'] - global_avg
            status = "OK" if abs(diff) < 3 else "REVIEW"
            print(f"  {region:25} | {diff:+.1f}% from avg | {status}")

    # FEATURE DISTRIBUTION
    print("\n" + "-" * 50)
    print("FEATURE SHAPE DISTRIBUTION BY REGION")
    print("-" * 50)

    for feature in ['eye_shape', 'nose_shape', 'lip_shape']:
        print(f"\n{feature.upper()}:")
        for region, celebs in sorted(by_region.items()):
            shapes = {}
            for c in celebs:
                if 'feature_shapes' in c:
                    fs = c['feature_shapes']
                    if feature == 'eye_shape':
                        shape = fs.get('eyes', {}).get('shape', 'unknown')
                    elif feature == 'nose_shape':
                        shape = fs.get('nose', {}).get('shape', 'unknown')
                    elif feature == 'lip_shape':
                        shape = fs.get('lips', {}).get('shape', 'unknown')
                    shapes[shape] = shapes.get(shape, 0) + 1

            if shapes:
                dist = ", ".join([f"{k}:{v}" for k, v in sorted(shapes.items(), key=lambda x: -x[1])])
                print(f"  {region:22} | {dist}")

    # CONTRAST/UNDERTONE
    print("\n" + "-" * 50)
    print("CONTRAST & UNDERTONE DISTRIBUTION")
    print("-" * 50)

    for metric in ['contrast', 'undertone']:
        print(f"\n{metric.upper()}:")
        for region, celebs in sorted(by_region.items()):
            values = {}
            for c in celebs:
                if 'scores' in c:
                    val = c['scores'].get(metric, 'unknown')
                    values[val] = values.get(val, 0) + 1

            if values:
                dist = ", ".join([f"{k}:{v}" for k, v in sorted(values.items(), key=lambda x: -x[1])])
                print(f"  {region:22} | {dist}")

    # Save report
    report_path = RESULTS_DIR / "bias_report.txt"
    print(f"\n\nReport saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download and analyze images')
    parser.add_argument('--report', action='store_true', help='Generate report from results')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of celebrities')
    parser.add_argument('--region', type=str, default=None, help='Test specific region only')
    args = parser.parse_args()

    if args.report:
        generate_report()
    elif args.download:
        run_batch_test(limit=args.limit)
    else:
        print("Usage:")
        print("  python download_and_test.py --download          # Run full test")
        print("  python download_and_test.py --download --limit 50  # Test first 50")
        print("  python download_and_test.py --report            # Generate report")
