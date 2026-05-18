import os
from pathlib import Path

from .timeline_audio_config import AUDIO_EXTENSIONS


def list_audio(root, recursive=True):
    root = Path(root)
    results = []
    if not root.is_dir():
        return results

    walker = os.walk(root) if recursive else [(root, [], [path.name for path in root.iterdir() if path.is_file()])]
    for dirpath, _, filenames in walker:
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            rel = path.relative_to(root).as_posix()
            stat = path.stat() if path.exists() else None
            results.append(
                {
                    "filename": rel,
                    "size": stat.st_size if stat else 0,
                    "mtime": stat.st_mtime if stat else 0,
                }
            )
    return sorted(results, key=lambda item: item["filename"].lower())
