import hashlib
import os
from pathlib import Path

from PIL import Image, ImageOps

from .timeline_image_config import IMAGE_EXTENSIONS, THUMB_CACHE_DIR


def load_rgb_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def list_images(root, recursive=True):
    root = Path(root)
    results = []
    if not root.is_dir():
        return results

    walker = os.walk(root) if recursive else [(root, [], [path.name for path in root.iterdir() if path.is_file()])]
    for dirpath, _, filenames in walker:
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel = path.relative_to(root).as_posix()
            width = height = 0
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                pass

            results.append(
                {
                    "filename": rel,
                    "width": width,
                    "height": height,
                    "mtime": path.stat().st_mtime if path.exists() else 0,
                }
            )
    return sorted(results, key=lambda item: item["filename"].lower())


def thumbnail_path(image_path, max_size=320):
    image_path = Path(image_path)
    mtime = image_path.stat().st_mtime if image_path.exists() else 0
    key = hashlib.sha256(f"{image_path}:{mtime}:{max_size}".encode("utf-8")).hexdigest()
    return THUMB_CACHE_DIR / f"{key}.webp"


def make_thumbnail(image_path, max_size=320):
    THUMB_CACHE_DIR.mkdir(exist_ok=True)
    out = thumbnail_path(image_path, max_size)
    if out.exists():
        return out

    image = load_rgb_image(image_path)
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image.save(out, "WEBP", quality=90, method=4)
    return out
