"""Character reference parsing helpers for LTX Director."""

from __future__ import annotations

import re
from typing import Any, Mapping


SUPPORTED_REFERENCE_KIND = "character"
REFERENCE_TAG_RE = re.compile(r"@(?P<label>image[1-9]\d*):(?P<kind>[A-Za-z][A-Za-z0-9_-]*)")


def _safe_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reference_label(value: Any, fallback_index: int) -> str:
    label = str(value or "").strip().lower()
    if re.fullmatch(r"image[1-9]\d*", label):
        return label
    return f"image{fallback_index + 1}"


def normalize_reference_images(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    references: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue

        label = _reference_label(item.get("label"), len(references))
        kind = str(item.get("kind") or SUPPORTED_REFERENCE_KIND).strip().lower()
        if kind != SUPPORTED_REFERENCE_KIND:
            continue

        ref = {
            "id": str(item.get("id") or label),
            "label": label,
            "kind": SUPPORTED_REFERENCE_KIND,
            "enabled": _safe_bool(item.get("enabled"), True),
            "strength": _safe_float(item.get("strength"), 1.0),
        }

        for key in (
            "imageFolderAlias",
            "imageFile",
            "imageB64",
            "thumb_url",
            "image_url",
            "filename",
            "fileName",
        ):
            if key in item:
                ref[key] = item.get(key)

        references.append(ref)

    return references


def parse_reference_tags(prompt: Any) -> list[dict[str, str]]:
    text = str(prompt or "")
    tags: list[dict[str, str]] = []
    for match in REFERENCE_TAG_RE.finditer(text):
        label = match.group("label").lower()
        kind = match.group("kind").lower()
        tags.append(
            {
                "label": label,
                "kind": kind,
                "token": match.group(0),
                "supported": kind == SUPPORTED_REFERENCE_KIND,
            }
        )
    return tags


def strip_reference_tags(prompt: Any) -> str:
    text = str(prompt or "")
    stripped = REFERENCE_TAG_RE.sub(" ", text)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"[ \t]+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"([([{])[ \t]+", r"\1", stripped)
    return stripped.strip()


def strip_reference_tags_from_prompt_list(local_prompts: Any) -> str:
    if local_prompts is None:
        return ""
    return " | ".join(strip_reference_tags(part) for part in str(local_prompts).split("|"))


def build_segment_reference_usage(timeline: Any, duration_frames: Any = None) -> list[dict[str, Any]]:
    if not isinstance(timeline, Mapping):
        return []

    try:
        duration = int(duration_frames) if duration_frames is not None else None
    except (TypeError, ValueError):
        duration = None

    references = normalize_reference_images(timeline.get("referenceImages", []))
    reference_by_label = {
        ref["label"]: ref
        for ref in references
        if ref.get("enabled", True)
    }

    usage: list[dict[str, Any]] = []
    for segment in timeline.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        try:
            start = int(segment.get("start", 0))
        except (TypeError, ValueError):
            start = 0
        if duration is not None and start >= duration:
            continue

        tags = parse_reference_tags(segment.get("prompt", ""))
        matched = []
        unknown = []
        unsupported = []
        seen = set()
        for tag in tags:
            if not tag["supported"]:
                unsupported.append(tag)
                continue
            ref = reference_by_label.get(tag["label"])
            if ref is None:
                unknown.append(tag)
                continue
            if ref["label"] in seen:
                continue
            seen.add(ref["label"])
            matched.append(ref)

        usage.append(
            {
                "segment_id": segment.get("id"),
                "start": start,
                "prompt": str(segment.get("prompt") or ""),
                "clean_prompt": strip_reference_tags(segment.get("prompt", "")),
                "references": matched,
                "unknown_tags": unknown,
                "unsupported_tags": unsupported,
            }
        )

    return usage


def build_reference_guide_specs(timeline: Any, duration_frames: Any = None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for usage in build_segment_reference_usage(timeline, duration_frames):
        for ref in usage.get("references", []):
            spec = dict(ref)
            spec["segment_id"] = usage.get("segment_id")
            spec["insert_frame"] = usage.get("start", 0)
            specs.append(spec)
    return specs


def reference_usage_errors(usages: list[dict[str, Any]]) -> dict[str, list[str]]:
    unknown = sorted(
        {
            tag["token"]
            for usage in usages
            for tag in usage.get("unknown_tags", [])
        }
    )
    unsupported = sorted(
        {
            tag["token"]
            for usage in usages
            for tag in usage.get("unsupported_tags", [])
        }
    )
    return {"unknown": unknown, "unsupported": unsupported}
