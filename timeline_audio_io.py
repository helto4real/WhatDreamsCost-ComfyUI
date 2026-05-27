import os
from pathlib import Path

import av

try:
    from .timeline_audio_config import AUDIO_EXTENSIONS
except ImportError:
    from timeline_audio_config import AUDIO_EXTENSIONS


def audio_duration_seconds(path):
    try:
        with av.open(path) as container:
            if container.duration is not None:
                return max(0.0, float(container.duration / av.time_base))
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if audio_streams:
                stream = audio_streams[0]
                if stream.duration is not None and stream.time_base is not None:
                    return max(0.0, float(stream.duration * stream.time_base))
    except Exception:
        return None
    return None


def list_audio(root, recursive=True, include_duration=False):
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
            item = {
                "filename": rel,
                "size": stat.st_size if stat else 0,
                "mtime": stat.st_mtime if stat else 0,
            }
            if include_duration:
                item["duration_seconds"] = audio_duration_seconds(path)
            results.append(item)
    return sorted(results, key=lambda item: item["filename"].lower())
