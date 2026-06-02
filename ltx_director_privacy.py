"""Privacy-aware LTX Director input normalization."""

from __future__ import annotations

import json
from typing import Any, Mapping

try:
    from .privacy import PrivacyError, decrypt_state, is_encrypted_payload
    from .ltx_director_references import normalize_reference_images, strip_reference_tags
except ImportError:  # Allows running tests from the repository root.
    from privacy import PrivacyError, decrypt_state, is_encrypted_payload
    from ltx_director_references import normalize_reference_images, strip_reference_tags


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_timeline(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        parsed = dict(value)
    elif isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
            parsed = dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            parsed = {}
    else:
        parsed = {}
    return {
        "segments": [dict(item) for item in parsed.get("segments", []) if isinstance(item, Mapping)],
        "audioSegments": [dict(item) for item in parsed.get("audioSegments", []) if isinstance(item, Mapping)],
        "referenceImages": normalize_reference_images(parsed.get("referenceImages", [])),
    }


def timeline_to_json(timeline: Any) -> str:
    return json.dumps(_parse_timeline(timeline), ensure_ascii=False, separators=(",", ":"))


def derive_timeline_outputs(timeline: Any, duration_frames: Any) -> dict[str, str]:
    parsed = _parse_timeline(timeline)
    sorted_segments = sorted(parsed["segments"], key=lambda seg: _as_int(seg.get("start"), 0))
    duration = max(1, _as_int(duration_frames, 1))

    contiguous_lengths: list[int] = []
    contiguous_prompts: list[str] = []
    current_cursor = 0
    pending_gap = 0

    for seg in sorted_segments:
        start = _as_int(seg.get("start"), 0)
        length = max(0, _as_int(seg.get("length"), 0))
        if start >= duration:
            break

        if start > current_cursor:
            gap_length = min(start, duration) - current_cursor
            if contiguous_lengths:
                contiguous_lengths[-1] += gap_length
            else:
                pending_gap += gap_length

        clipped_end = min(start + length, duration)
        clipped_length = max(0, clipped_end - start)
        contiguous_lengths.append(clipped_length + pending_gap)

        prompt = str(seg.get("prompt") or "")
        if seg.get("type") == "source_video" and not prompt.strip():
            for candidate in sorted_segments:
                candidate_start = _as_int(candidate.get("start"), 0)
                candidate_prompt = str(candidate.get("prompt") or "")
                if candidate is not seg and candidate_start >= start + length and candidate_prompt.strip():
                    prompt = candidate_prompt
                    break
        contiguous_prompts.append(strip_reference_tags(prompt))

        pending_gap = 0
        current_cursor = start + length

    clamped_cursor = min(current_cursor, duration)
    if contiguous_lengths and clamped_cursor < duration:
        contiguous_lengths[-1] += duration - clamped_cursor

    guide_strengths = []
    for seg in sorted_segments:
        if seg.get("type") == "text":
            continue
        try:
            guide_strengths.append(f"{float(seg.get('guideStrength', 1.0)):.2f}")
        except (TypeError, ValueError):
            guide_strengths.append("1.00")

    return {
        "local_prompts": " | ".join(contiguous_prompts),
        "segment_lengths": ",".join(str(length) for length in contiguous_lengths),
        "guide_strength": ",".join(guide_strengths),
    }


def resolve_ltx_director_inputs(
    *,
    global_prompt: Any,
    timeline_data: Any,
    local_prompts: Any,
    segment_lengths: Any,
    guide_strength: Any,
    duration_frames: Any,
    privacy_mode: Any = False,
    privacy_payload: Any = "",
    privacy_base_dir: Any = None,
) -> dict[str, str]:
    if not _as_bool(privacy_mode):
        return {
            "global_prompt": str(global_prompt or ""),
            "timeline_data": str(timeline_data or ""),
            "local_prompts": str(local_prompts or ""),
            "segment_lengths": str(segment_lengths or ""),
            "guide_strength": str(guide_strength or ""),
        }

    if not is_encrypted_payload(privacy_payload):
        raise PrivacyError("Privacy mode is enabled, but no encrypted LTX Director payload is available.")

    state = decrypt_state(privacy_payload, privacy_base_dir)
    timeline = _parse_timeline(state.get("timeline", {}))
    derived = derive_timeline_outputs(timeline, duration_frames)
    return {
        "global_prompt": str(state.get("global_prompt") or ""),
        "timeline_data": timeline_to_json(timeline),
        "local_prompts": derived["local_prompts"],
        "segment_lengths": derived["segment_lengths"],
        "guide_strength": derived["guide_strength"],
    }
