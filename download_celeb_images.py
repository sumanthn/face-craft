"""
Download celebrity images from IMDB for the carousel.
"""

import os
import requests
import time
from pathlib import Path

# Create images directory
IMAGES_DIR = Path(__file__).parent / "static" / "celebrities"
IMAGES_DIR.mkdir(exist_ok=True)

# Celebrity IMDB IDs and names
CELEBRITIES = [
    # East Asia
    ("nm0319700", "gong_li", "Gong Li", "China"),
    ("nm1256804", "song_hye_kyo", "Song Hye-kyo", "South Korea"),
    ("nm0913822", "ken_watanabe", "Ken Watanabe", "Japan"),
    ("nm1085859", "fan_bingbing", "Fan Bingbing", "China"),
    ("nm2632232", "lee_min_ho", "Lee Min-ho", "South Korea"),
    ("nm3222126", "gong_yoo", "Gong Yoo", "South Korea"),

    # South Asia
    ("nm0706787", "aishwarya_rai", "Aishwarya Rai", "India"),
    ("nm1231899", "priyanka_chopra", "Priyanka Chopra", "India"),
    ("nm1982597", "deepika_padukone", "Deepika Padukone", "India"),
    ("nm0451321", "shah_rukh_khan", "Shah Rukh Khan", "India"),
    ("nm0724536", "hrithik_roshan", "Hrithik Roshan", "India"),
    ("nm0451148", "aamir_khan", "Aamir Khan", "India"),

    # Middle East / North Africa
    ("nm2933757", "gal_gadot", "Gal Gadot", "Israel"),
    ("nm1785339", "rami_malek", "Rami Malek", "Egypt"),
    ("nm1291544", "golshifteh_farahani", "Golshifteh Farahani", "Iran"),
    ("nm0000204", "natalie_portman", "Natalie Portman", "Israel"),

    # Sub-Saharan Africa
    ("nm2143282", "lupita_nyongo", "Lupita Nyong'o", "Kenya"),
    ("nm0252961", "idris_elba", "Idris Elba", "UK/Sierra Leone"),
    ("nm0250809", "chiwetel_ejiofor", "Chiwetel Ejiofor", "Nigeria"),
    ("nm1659547", "john_boyega", "John Boyega", "UK/Nigeria"),
    ("nm1935086", "daniel_kaluuya", "Daniel Kaluuya", "UK/Uganda"),
    ("nm0147147", "naomi_campbell", "Naomi Campbell", "UK"),

    # Latin America
    ("nm0000161", "salma_hayek", "Salma Hayek", "Mexico"),
    ("nm1002641", "oscar_isaac", "Oscar Isaac", "Guatemala"),
    ("nm0305558", "gael_garcia_bernal", "Gael García Bernal", "Mexico"),
    ("nm0757855", "zoe_saldana", "Zoe Saldana", "Dominican Rep."),
    ("nm1869101", "ana_de_armas", "Ana de Armas", "Cuba"),
    ("nm0631625", "pedro_pascal", "Pedro Pascal", "Chile"),
    ("nm0000151", "penelope_cruz", "Penélope Cruz", "Spain"),

    # Europe
    ("nm0180411", "marion_cotillard", "Marion Cotillard", "France"),
    ("nm0586568", "mads_mikkelsen", "Mads Mikkelsen", "Denmark"),
    ("nm0000899", "monica_bellucci", "Monica Bellucci", "Italy"),
    ("nm0080720", "cate_blanchett", "Cate Blanchett", "Australia"),
    ("nm3053338", "margot_robbie", "Margot Robbie", "Australia"),
    ("nm0186505", "daniel_craig", "Daniel Craig", "UK"),
    ("nm0185819", "benedict_cumberbatch", "Benedict Cumberbatch", "UK"),
    ("nm0914612", "emma_watson", "Emma Watson", "UK"),
    ("nm0000234", "charlize_theron", "Charlize Theron", "South Africa"),

    # Pacific / Indigenous
    ("nm0597388", "jason_momoa", "Jason Momoa", "Hawaii"),
    ("nm0169806", "taika_waititi", "Taika Waititi", "New Zealand"),
    ("nm0186469", "cliff_curtis", "Cliff Curtis", "New Zealand"),
    ("nm0597682", "dwayne_johnson", "Dwayne Johnson", "Polynesian"),

    # Mixed Heritage / Diverse
    ("nm3918035", "zendaya", "Zendaya", "Mixed Heritage"),
    ("nm0000206", "keanu_reeves", "Keanu Reeves", "Mixed Heritage"),
    ("nm0000288", "halle_berry", "Halle Berry", "Mixed Heritage"),
    ("nm0680983", "dev_patel", "Dev Patel", "UK/India"),
    ("nm0000706", "michelle_yeoh", "Michelle Yeoh", "Malaysia"),
    ("nm0205626", "viola_davis", "Viola Davis", "USA"),
    ("nm0000982", "denzel_washington", "Denzel Washington", "USA"),
    ("nm1620783", "alicia_vikander", "Alicia Vikander", "Sweden"),
]

def get_imdb_image(imdb_id: str) -> str:
    """Get the main profile image URL from IMDB."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    url = f"https://www.imdb.com/name/{imdb_id}/"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Find the primary image URL in the HTML
        html = response.text

        # Look for the poster/profile image
        # IMDB uses various patterns, try to find the main image
        import re

        # Pattern 1: Look for the primary image in JSON-LD
        json_match = re.search(r'"image":\s*"(https://[^"]+)"', html)
        if json_match:
            return json_match.group(1)

        # Pattern 2: Look for poster image
        poster_match = re.search(r'class="ipc-image"[^>]*src="(https://m\.media-amazon\.com/images/[^"]+)"', html)
        if poster_match:
            return poster_match.group(1)

        # Pattern 3: Any media-amazon image that looks like a headshot
        img_match = re.search(r'(https://m\.media-amazon\.com/images/M/[^"]+(?:_V1_|_CR)[^"]+\.jpg)', html)
        if img_match:
            return img_match.group(1)

    except Exception as e:
        print(f"  Error fetching {imdb_id}: {e}")

    return None


def resize_imdb_url(url: str, width: int = 400) -> str:
    """Resize IMDB image URL to desired width."""
    if not url:
        return url
    # IMDB URLs can be resized by modifying the _V1_ parameters
    import re
    # Replace sizing parameters to get desired size
    resized = re.sub(r'_V1_.*\.jpg', f'_V1_UX{width}_.jpg', url)
    return resized


def download_image(url: str, filepath: Path) -> bool:
    """Download image to file."""
    if filepath.exists():
        print(f"  Already exists: {filepath.name}")
        return True

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        if len(response.content) > 5000:  # Ensure it's a real image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  Downloaded: {filepath.name}")
            return True
    except Exception as e:
        print(f"  Download failed: {e}")

    return False


def main():
    print(f"\nDownloading {len(CELEBRITIES)} celebrity images from IMDB...\n")

    success = 0
    failed = []

    for imdb_id, filename, name, region in CELEBRITIES:
        print(f"Processing: {name} ({region})")

        filepath = IMAGES_DIR / f"{filename}.jpg"

        if filepath.exists():
            print(f"  Already exists")
            success += 1
            continue

        # Get image URL from IMDB
        img_url = get_imdb_image(imdb_id)

        if img_url:
            # Resize to 400px width
            img_url = resize_imdb_url(img_url, 400)

            if download_image(img_url, filepath):
                success += 1
            else:
                failed.append(name)
        else:
            print(f"  Could not find image URL")
            failed.append(name)

        # Be nice to IMDB
        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Downloaded: {success}/{len(CELEBRITIES)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Images saved to: {IMAGES_DIR}")

    # Generate JS array for index.html
    print(f"\n{'='*50}")
    print("Celebrity array for index.html:\n")
    print("const celebrities = [")
    for imdb_id, filename, name, region in CELEBRITIES:
        filepath = IMAGES_DIR / f"{filename}.jpg"
        if filepath.exists():
            print(f'    {{ name: "{name}", region: "{region}", img: "/static/celebrities/{filename}.jpg" }},')
    print("];")


if __name__ == "__main__":
    main()
