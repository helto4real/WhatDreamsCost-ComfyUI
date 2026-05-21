import hashlib
import io
import json
import os
from pathlib import Path
import shutil

from PIL import Image, ImageOps

try:
    from .privacy import decrypt_bytes, encrypt_bytes
    from .timeline_image_config import IMAGE_EXTENSIONS, THUMB_CACHE_DIR
except ImportError:
    from privacy import decrypt_bytes, encrypt_bytes
    from timeline_image_config import IMAGE_EXTENSIONS, THUMB_CACHE_DIR


THUMBNAIL_CACHE_PURPOSE = "timeline-thumbnail-cache"


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


def thumbnail_path(image_path, max_size=320, privacy_mode=False, cache_dir=None):
    image_path = Path(image_path)
    mtime = image_path.stat().st_mtime if image_path.exists() else 0
    mode = "private" if privacy_mode else "plain"
    key = hashlib.sha256(f"{image_path}:{mtime}:{max_size}:{mode}".encode("utf-8")).hexdigest()
    suffix = ".webp.enc" if privacy_mode else ".webp"
    return (Path(cache_dir) if cache_dir is not None else THUMB_CACHE_DIR) / f"{key}{suffix}"


def clear_thumbnail_cache(cache_dir=None):
    root = Path(cache_dir) if cache_dir is not None else THUMB_CACHE_DIR
    root.mkdir(exist_ok=True)
    resolved_root = root.resolve()
    for child in root.iterdir():
        resolved_child = child.resolve()
        if resolved_root != resolved_child and resolved_root not in resolved_child.parents:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _thumbnail_bytes(image_path, max_size):
    image = load_rgb_image(image_path)
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=90, method=4)
    return buffer.getvalue()


def _write_json_private(path, payload):
    path.parent.mkdir(exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    tmp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def make_thumbnail(image_path, max_size=320, privacy_mode=False, privacy_base_dir=None, cache_dir=None):
    cache_root = Path(cache_dir) if cache_dir is not None else THUMB_CACHE_DIR
    cache_root.mkdir(exist_ok=True)
    out = thumbnail_path(image_path, max_size, privacy_mode=privacy_mode, cache_dir=cache_root)
    if out.exists():
        if privacy_mode:
            return decrypt_bytes(out.read_text(encoding="utf-8"), THUMBNAIL_CACHE_PURPOSE, privacy_base_dir)
        return out

    thumbnail = _thumbnail_bytes(image_path, max_size)
    if privacy_mode:
        _write_json_private(out, encrypt_bytes(thumbnail, THUMBNAIL_CACHE_PURPOSE, privacy_base_dir))
        return thumbnail

    out.write_bytes(thumbnail)
    return out
