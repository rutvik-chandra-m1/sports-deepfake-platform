"""
Downloads real, openly-licensed sports photographs from Wikimedia Commons
for the R2 sports-specific "real" supplement.

Commons hosts only public-domain or freely-licensed media by policy, but we
still verify and record the exact license per file (required for correct
attribution of CC-BY/CC-BY-SA images, and for the dataset card) rather than
trusting that blindly. Only CC0 / Public Domain / CC-BY / CC-BY-SA are kept;
anything else (or missing license metadata) is skipped.

Usage:
    python fetch_wikimedia_sports.py --per-category 15 --out ../../datasets/sports_real
"""

import argparse
import csv
import hashlib
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
# Wikimedia's API etiquette policy requires a descriptive User-Agent
# identifying the tool and a contact point.
HEADERS = {
    "User-Agent": (
        "SportsDeepfakePlatform-DatasetBuilder/0.1 "
        "(final-year academic project; https://github.com/) "
        "python-requests"
    )
}

ACCEPTED_LICENSES = {"cc0", "pd", "public domain", "cc-by", "cc-by-sa"}

# A spread of sports/categories for variety rather than one narrow niche.
# "Track and field athletes" was tried first but only contains subcategories,
# no direct file members -- "Sprinters" is the verified working substitute.
CATEGORIES = [
    "Category:Football players",
    "Category:Basketball players",
    "Category:Tennis players",
    "Category:Sprinters",
    "Category:Swimmers",
    "Category:Volleyball players",
    "Category:Cricketers",
]

REQUEST_DELAY_SECONDS = 0.5


def _get(params: dict) -> dict:
    params = {**params, "format": "json"}
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_category_files(category: str, limit: int) -> list[str]:
    data = _get(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": limit,
        }
    )
    members = data.get("query", {}).get("categorymembers", [])
    return [m["title"] for m in members]


def get_image_info(titles: list[str]) -> dict:
    """Batched imageinfo lookup (URL + license metadata) for up to 50 titles."""
    data = _get(
        {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
        }
    )
    return data.get("query", {}).get("pages", {})


def _normalize_license(extmetadata: dict) -> str | None:
    license_short = extmetadata.get("LicenseShortName", {}).get("value", "")
    return license_short.strip().lower() or None


def fetch_category(category: str, count: int, out_dir: Path, manifest_rows: list) -> int:
    logger.info("Listing files in %s", category)
    titles = list_category_files(category, limit=min(count * 3, 100))
    if not titles:
        logger.warning("No files found in %s", category)
        return 0

    time.sleep(REQUEST_DELAY_SECONDS)
    pages = get_image_info(titles)

    downloaded = 0
    for page in pages.values():
        if downloaded >= count:
            break
        imageinfo = page.get("imageinfo")
        if not imageinfo:
            continue
        info = imageinfo[0]
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue  # skip SVG/TIFF/etc -- not natural photographs

        extmetadata = info.get("extmetadata", {})
        license_key = _normalize_license(extmetadata)
        if license_key not in ACCEPTED_LICENSES:
            logger.debug("Skipping %s: license '%s' not in accepted set", page.get("title"), license_key)
            continue

        url = info["url"]
        width, height = info.get("width", 0), info.get("height", 0)
        if width < 300 or height < 300:
            continue  # too small to be a useful training image

        title = page.get("title", "unknown")
        artist = extmetadata.get("Artist", {}).get("value", "unknown")
        credit = extmetadata.get("Credit", {}).get("value", "")
        license_url = extmetadata.get("LicenseUrl", {}).get("value", "")

        ext = ".jpg" if mime == "image/jpeg" else ".png"
        file_id = hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
        dest = out_dir / f"wikimedia_{file_id}{ext}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Download failed for %s: %s", title, exc)
            continue

        dest.write_bytes(resp.content)
        downloaded += 1
        manifest_rows.append(
            {
                "filename": dest.name,
                "source": "wikimedia_commons",
                "source_title": title,
                "source_url": url,
                "license": license_key,
                "license_url": license_url,
                "artist": artist,
                "credit": credit,
                "category": category,
                "width": width,
                "height": height,
            }
        )
        logger.info("  [%d/%d] %s (%s, %dx%d)", downloaded, count, dest.name, license_key, width, height)
        time.sleep(REQUEST_DELAY_SECONDS)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=15)
    parser.add_argument("--out", type=str, default="../../datasets/sports_real")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    total = 0
    for category in CATEGORIES:
        got = fetch_category(category, args.per_category, out_dir, manifest_rows)
        total += got
        logger.info("%s -> %d images", category, got)

    manifest_path = out_dir / "attribution.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "source", "source_title", "source_url", "license",
                "license_url", "artist", "credit", "category", "width", "height",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info("Done: %d real sports photos in %s (attribution: %s)", total, out_dir, manifest_path)


if __name__ == "__main__":
    main()
