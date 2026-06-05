import logging
import json
import base64
import io as _io
import math
from collections import deque

import numpy as np
import torch
import av
from PIL import Image

import os
import folder_paths
import comfy.model_management

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, apply_patches
from .timeline_image_config import resolve_image_path
from .timeline_audio_config import resolve_audio_path
from .ltx_director_privacy import resolve_ltx_director_inputs
from .ltx_director_references import (
    build_reference_guide_specs,
    build_segment_reference_usage,
    normalize_reference_images,
    reference_usage_errors,
    strip_reference_tags_from_prompt_list,
)

log = logging.getLogger(__name__)

# Custom socket type shared with LTXSequencer
GuideData = io.Custom("GUIDE_DATA")


class LTXDirectorReferenceError(ValueError):
    pass

LTX_RESOLUTION_PRESETS = {
    "1:1": [
        (512, 512),
        (640, 640),
        (768, 768),
        (896, 896),
        (1024, 1024),
        (1088, 1088),
    ],
    "4:3": [
        (384, 512),
        (480, 640),
        (576, 768),
        (672, 896),
        (768, 1024),
        (1056, 1408),
    ],
    "3:2": [
        (384, 576),
        (512, 768),
        (576, 864),
        (704, 1056),
        (768, 1152),
        (1088, 1632),
    ],
    "16:9": [
        (320, 576),
        (448, 768),
        (512, 896),
        (576, 1024),
        (704, 1216),
        (1088, 1920),
    ],
}

LTX_QUALITY_TIERS = [
    "1 - fast samples",
    "2 - fast and ok",
    "3 - reasonable",
    "4 - better details",
    "5 - really good",
    "6 - LTX 2.3 native",
]

AUDIO_NORMALIZE_TARGET_RMS = 10 ** (-18.0 / 20.0)
AUDIO_NORMALIZE_PEAK_CEILING = 10 ** (-1.0 / 20.0)
AUDIO_NORMALIZE_MAX_GAIN = 2.0
AUDIO_NORMALIZE_EPSILON = 1e-8


def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image from the ComfyUI input folder (if imageFile provided) or fallback to base64
    to a ComfyUI-style image tensor of shape [1, H, W, 3], float32 in [0, 1]."""
    if seg.get("imageFolderAlias") and seg.get("imageFile"):
        try:
            img = Image.open(resolve_image_path(seg["imageFolderAlias"], seg["imageFile"])).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception as exc:
            log.warning("[PromptRelay] Could not load timeline browser image %s/%s: %s", seg.get("imageFolderAlias"), seg.get("imageFile"), exc)

    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _reference_image_segment(ref: dict) -> dict:
    seg = dict(ref)
    if not seg.get("imageFile") and seg.get("filename"):
        seg["imageFile"] = seg.get("filename")
    if not seg.get("imageFile") and seg.get("fileName"):
        seg["imageFile"] = seg.get("fileName")
    if not seg.get("imageFile") and seg.get("image_file"):
        seg["imageFile"] = seg.get("image_file")
    return seg


def _load_video_tail_tensor(seg: dict, frame_count: int) -> torch.Tensor:
    """Decode the final `frame_count` frames from an uploaded source video.
    Returns a ComfyUI-style image tensor of shape [N, H, W, 3]."""
    try:
        frame_count = int(frame_count or 9)
    except (TypeError, ValueError):
        frame_count = 9
    frame_count = max(1, min(65, frame_count))
    video_file = seg.get("videoFile")
    if not video_file:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    file_path = os.path.join(folder_paths.get_input_directory(), video_file)
    if not os.path.exists(file_path):
        log.warning("[PromptRelay] Source video not found: %s", video_file)
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    frames = deque(maxlen=frame_count)
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                frames.append(torch.from_numpy(arr))
    except Exception as exc:
        log.warning("[PromptRelay] Could not decode source video %s: %s", video_file, exc)
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    return torch.stack(list(frames), dim=0)


def _source_video_path(seg: dict) -> str | None:
    video_file = seg.get("videoFile")
    if not video_file:
        return None
    file_path = os.path.join(folder_paths.get_input_directory(), video_file)
    if not os.path.exists(file_path):
        log.warning("[PromptRelay] Source video not found: %s", video_file)
        return None
    return file_path


def _stream_fps(stream, fallback: float) -> float:
    for attr in ("average_rate", "base_rate", "guessed_rate"):
        rate = getattr(stream, attr, None)
        if rate:
            try:
                value = float(rate)
                if value > 0:
                    return value
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    return float(fallback or 24.0)


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clamp_audio_volume(value) -> float:
    try:
        volume = float(value)
    except (TypeError, ValueError):
        volume = 1.0
    return max(0.0, min(2.0, volume))


def _normalize_audio_waveform(waveform: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    if not enabled or not torch.is_tensor(waveform) or waveform.numel() == 0:
        return waveform

    analysis = torch.nan_to_num(waveform.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    peak = torch.max(torch.abs(analysis)).item()
    if peak <= AUDIO_NORMALIZE_EPSILON:
        return waveform

    rms = torch.sqrt(torch.mean(analysis * analysis)).item()
    if rms <= AUDIO_NORMALIZE_EPSILON:
        return waveform

    rms_gain = AUDIO_NORMALIZE_TARGET_RMS / rms
    peak_gain = AUDIO_NORMALIZE_PEAK_CEILING / peak
    gain = min(rms_gain, peak_gain)
    if gain > 1.0:
        gain = min(gain, AUDIO_NORMALIZE_MAX_GAIN)

    normalized = waveform * gain
    final_analysis = torch.nan_to_num(normalized.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    final_peak = torch.max(torch.abs(final_analysis)).item()
    if final_peak > AUDIO_NORMALIZE_PEAK_CEILING:
        normalized = normalized * (AUDIO_NORMALIZE_PEAK_CEILING / final_peak)
    return normalized


def _empty_audio(duration_seconds: float = 1.0, sample_rate: int = 44100, channels: int = 2) -> dict:
    total_samples = max(1, int(math.ceil(max(0.0, duration_seconds) * sample_rate)))
    return {
        "waveform": torch.zeros((1, channels, total_samples), dtype=torch.float32),
        "sample_rate": sample_rate,
    }


def _decode_source_video_audio(file_path: str, duration_seconds: float, target_sr: int = 44100) -> dict:
    try:
        clip_frames = []
        with av.open(file_path) as container:
            if not container.streams.audio:
                return _empty_audio(duration_seconds, target_sr)

            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            resampler = av.AudioResampler(
                format="fltp",
                layout="stereo",
                rate=target_sr,
            )

            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    clip_frames.append(torch.from_numpy(resampled_frame.to_ndarray()))

            for resampled_frame in resampler.resample(None):
                clip_frames.append(torch.from_numpy(resampled_frame.to_ndarray()))

        if not clip_frames:
            return _empty_audio(duration_seconds, target_sr)

        waveform = torch.cat(clip_frames, dim=1).to(torch.float32)
        expected_samples = max(1, int(math.ceil(max(0.0, duration_seconds) * target_sr)))
        if waveform.shape[1] < expected_samples:
            pad = torch.zeros((waveform.shape[0], expected_samples - waveform.shape[1]), dtype=waveform.dtype)
            waveform = torch.cat((waveform, pad), dim=1)
        elif waveform.shape[1] > expected_samples:
            waveform = waveform[:, :expected_samples]

        return {"waveform": waveform.unsqueeze(0), "sample_rate": target_sr}
    except Exception as exc:
        log.warning("[PromptRelay] Could not decode source video audio %s: %s", os.path.basename(file_path), exc)
        return _empty_audio(duration_seconds, target_sr)


def _load_source_video_outputs(
    seg: dict | None,
    target_w: int,
    target_h: int,
    resize_method: str,
    divisible_by: int,
    fallback_frame_rate: float,
) -> tuple[torch.Tensor, dict, float, int]:
    """Decode the full uploaded source video for downstream stitching.
    Returns stitch-ready image frames resized like timeline guide images, source audio, source FPS, and frame count."""
    fallback_fps = float(fallback_frame_rate or 24.0)
    empty_images = torch.zeros((1, max(1, target_h), max(1, target_w), 3), dtype=torch.float32)
    if not seg:
        return empty_images, _empty_audio(1.0), fallback_fps, 0

    file_path = _source_video_path(seg)
    if not file_path:
        return empty_images, _empty_audio(1.0), fallback_fps, 0

    frames = []
    fps = fallback_fps
    try:
        with av.open(file_path) as container:
            if not container.streams.video:
                return empty_images, _empty_audio(1.0), fallback_fps, 0

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = _stream_fps(stream, fallback_fps)

            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                frames.append(torch.from_numpy(arr))
    except Exception as exc:
        log.warning("[PromptRelay] Could not decode full source video %s: %s", seg.get("videoFile"), exc)
        return empty_images, _empty_audio(1.0), fallback_fps, 0

    if not frames:
        return empty_images, _empty_audio(1.0), fps, 0

    source_images = torch.stack(frames, dim=0)
    frame_count = int(source_images.shape[0])
    duration_seconds = frame_count / fps if fps > 0 else 1.0

    source_images = _resize_image_frames(
        source_images,
        max(1, target_w),
        max(1, target_h),
        resize_method,
        divisible_by,
    )
    source_audio = _decode_source_video_audio(file_path, duration_seconds)
    return source_images, source_audio, fps, frame_count


def _ltx_preset_dimensions(aspect_ratio: str, orientation: str, quality_tier: str) -> tuple[int, int]:
    presets = LTX_RESOLUTION_PRESETS.get(aspect_ratio, LTX_RESOLUTION_PRESETS["16:9"])
    try:
        tier_index = int(str(quality_tier).split(" - ", 1)[0]) - 1
    except (TypeError, ValueError):
        tier_index = len(presets) - 1
    tier_index = max(0, min(tier_index, len(presets) - 1))

    short_side, long_side = presets[tier_index]
    if aspect_ratio == "1:1":
        return short_side, short_side
    if orientation == "portrait":
        return short_side, long_side
    return long_side, short_side


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    """Resize a [1, H, W, 3] float32 tensor to target dimensions using the given method,
    then snap the final dimensions to be divisible by `divisible_by`."""
    from PIL import Image as _PilImage
    import torchvision.transforms.functional as TF

    def snap(val, div):
        return max(div, (val // div) * div)

    def fit_size(max_w, max_h):
        ratio = min(max_w / src_w, max_h / src_h)
        new_w = max(1, min(max_w, int(round(src_w * ratio))))
        new_h = max(1, min(max_h, int(round(src_h * ratio))))
        return new_w, new_h

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    img_np = (tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = _PilImage.fromarray(img_np)
    src_w, src_h = pil.size

    if method == "stretch to fit":
        resized = pil.resize((tw, th), _PilImage.LANCZOS)

    elif method == "maintain aspect ratio":
        new_w, new_h = fit_size(tw, th)
        resized = pil.resize((new_w, new_h), _PilImage.LANCZOS)

    elif method == "pad":
        new_w, new_h = fit_size(tw, th)
        inner = pil.resize((new_w, new_h), _PilImage.LANCZOS)
        resized = _PilImage.new("RGB", (tw, th), (0, 0, 0))
        resized.paste(inner, ((tw - new_w) // 2, (th - new_h) // 2))

    elif method == "crop":
        ratio = max(tw / src_w, th / src_h)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        inner = pil.resize((new_w, new_h), _PilImage.LANCZOS)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner.crop((left, top, left + tw, top + th))

    else:
        resized = pil.resize((tw, th), _PilImage.LANCZOS)

    arr = np.array(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _resize_image_frames(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    if tensor.shape[0] <= 1:
        return _resize_image(tensor, target_w, target_h, method, divisible_by)
    return torch.cat([
        _resize_image(tensor[i:i + 1], target_w, target_h, method, divisible_by)
        for i in range(tensor.shape[0])
    ], dim=0)


def _resize_reference_image_frames(
    tensor: torch.Tensor,
    target_w: int,
    target_h: int,
    derived_w: int,
    derived_h: int,
    use_input_image_size: bool,
    divisible_by: int,
) -> torch.Tensor:
    src_h, src_w = tensor.shape[1], tensor.shape[2]
    return _resize_image_frames(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)


def _resize_reference_guide_frames(
    tensor: torch.Tensor,
    target_w: int,
    target_h: int,
    derived_w: int,
    derived_h: int,
    use_input_image_size: bool,
    divisible_by: int,
) -> torch.Tensor:
    guide_w = derived_w if use_input_image_size and derived_w > 0 else target_w
    guide_h = derived_h if use_input_image_size and derived_h > 0 else target_h
    return _resize_image_frames(tensor, guide_w, guide_h, "pad", divisible_by)


def _stack_timeline_identity_images(
    tensors: list[torch.Tensor],
    target_w: int,
    target_h: int,
    derived_w: int,
    derived_h: int,
    divisible_by: int,
) -> torch.Tensor | None:
    if not tensors:
        return None

    first_h, first_w = tensors[0].shape[1], tensors[0].shape[2]
    if all(tensor.shape[1] == first_h and tensor.shape[2] == first_w for tensor in tensors):
        return torch.cat(tensors, dim=0)

    fallback_w = derived_w if derived_w > 0 else target_w
    fallback_h = derived_h if derived_h > 0 else target_h
    return torch.cat(
        [
            _resize_image_frames(tensor, fallback_w, fallback_h, "pad", divisible_by)
            for tensor in tensors
        ],
        dim=0,
    )


def _dedupe_reference_specs(reference_specs: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    index_by_key: dict[str, int] = {}
    for spec in reference_specs:
        key = str(spec.get("label") or spec.get("id") or "").strip().lower()
        if not key:
            key = str(len(deduped))

        segment_id = spec.get("segment_id")
        if key in index_by_key:
            existing = deduped[index_by_key[key]]
            ids = [
                part
                for part in str(existing.get("segment_id") or "").split(",")
                if part
            ]
            if segment_id is not None and str(segment_id) not in ids:
                ids.append(str(segment_id))
            if ids:
                existing["segment_id"] = ",".join(ids)
            continue

        entry = dict(spec)
        if segment_id is not None:
            entry["segment_id"] = str(segment_id)
        deduped.append(entry)
        index_by_key[key] = len(deduped) - 1
    return deduped


def _crop_latent_to_frame_count(latent, frame_count, hidden_reference_count=0):
    try:
        metadata_target = int(frame_count)
    except (TypeError, ValueError):
        return latent
    try:
        hidden_reference_count = max(0, int(hidden_reference_count or 0))
    except (TypeError, ValueError):
        hidden_reference_count = 0
    if metadata_target <= 0 or not isinstance(latent, dict):
        return latent

    cropped = latent.copy()

    def tensor_frame_count(tensor):
        if not torch.is_tensor(tensor):
            return None
        if tensor.ndim == 5:
            return int(tensor.shape[2])
        if tensor.ndim in (3, 4):
            return int(tensor.shape[0])
        return None

    def crop_tensor(tensor):
        if not torch.is_tensor(tensor):
            return tensor
        current_frames = tensor_frame_count(tensor)
        if current_frames is None:
            return tensor
        keep_frames = min(target_frames, current_frames)
        if tensor.ndim == 5:
            try:
                return torch.narrow(tensor, 2, 0, keep_frames)
            except Exception:
                return tensor[:, :, :keep_frames, :, :]
        if tensor.ndim in (3, 4):
            try:
                return torch.narrow(tensor, 0, 0, keep_frames)
            except Exception:
                return tensor[:keep_frames]
        return tensor

    def first_video_stream(value):
        if getattr(value, "is_nested", False):
            try:
                streams = list(value.unbind())
            except Exception:
                return None
            return streams[0] if streams else None
        return value

    def crop_value(value):
        if getattr(value, "is_nested", False):
            try:
                streams = list(value.unbind())
            except Exception:
                return value
            if not streams:
                return value
            streams[0] = crop_tensor(streams[0])
            try:
                return type(value)(streams)
            except Exception:
                return value
        return crop_tensor(value)

    before_frames = tensor_frame_count(first_video_stream(cropped.get("samples")))
    candidate_targets = [metadata_target]
    if hidden_reference_count > 0 and before_frames is not None:
        tail_count_target = before_frames - hidden_reference_count
        if tail_count_target > 0:
            candidate_targets.append(tail_count_target)
    target_frames = min(candidate_targets)

    if "samples" in cropped:
        cropped["samples"] = crop_value(cropped["samples"])
    if "noise_mask" in cropped:
        cropped["noise_mask"] = crop_value(cropped["noise_mask"])
    after_frames = tensor_frame_count(first_video_stream(cropped.get("samples")))
    if before_frames is not None and after_frames is not None:
        if before_frames <= target_frames:
            log.warning(
                "[PromptRelay] LTX Director Crop Reference Tail received %d latent frames; "
                "metadata_target=%d, hidden_reference_count=%d, chosen_target=%d. "
                "No reference tail was removed.",
                before_frames,
                metadata_target,
                hidden_reference_count,
                target_frames,
            )
        else:
            log.info(
                "[PromptRelay] LTX Director Crop Reference Tail cropped video latent frames "
                "from %d to %d; metadata_target=%d, hidden_reference_count=%d, chosen_target=%d.",
                before_frames,
                after_frames,
                metadata_target,
                hidden_reference_count,
                target_frames,
            )
    return cropped


def _pad_latent_tail(latent, extra_latent_frames: int):
    if extra_latent_frames <= 0:
        return latent
    if not isinstance(latent, dict) or not torch.is_tensor(latent.get("samples")):
        raise ValueError("LTX Director hidden references need optional_latent to be a LATENT dict with tensor samples.")

    samples = latent["samples"]
    if samples.ndim != 5:
        raise ValueError(
            "LTX Director hidden references can only auto-pad 5D video latent samples "
            f"(got shape {tuple(samples.shape)})."
        )

    padded = latent.copy()
    pad_shape = list(samples.shape)
    pad_shape[2] = int(extra_latent_frames)
    padded["samples"] = torch.cat(
        [samples, torch.zeros(pad_shape, dtype=samples.dtype, device=samples.device)],
        dim=2,
    )

    noise_mask = latent.get("noise_mask")
    if torch.is_tensor(noise_mask) and noise_mask.ndim == 5:
        mask_shape = list(noise_mask.shape)
        mask_shape[2] = int(extra_latent_frames)
        padded["noise_mask"] = torch.cat(
            [noise_mask, torch.ones(mask_shape, dtype=noise_mask.dtype, device=noise_mask.device)],
            dim=2,
        )
    return padded


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Apply H.264 compression artefacts to a [1, H, W, 3] float32 tensor (ComfyUI image format).
    crf=0 means no compression. Uses PyAV to encode/decode a single frame in-memory."""
    if crf == 0:
        return tensor
    img = tensor[0]  # [H, W, 3]
    # Dimensions must be even for H.264
    h = (img.shape[0] // 2) * 2
    w = (img.shape[1] // 2) * 2
    img_np = (img[:h, :w] * 255.0).byte().cpu().numpy()  # uint8 [H, W, 3]

    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=1)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        frame = av.VideoFrame.from_ndarray(img_np, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = None
        for frame_r in container_r.decode(video=0):
            decoded = frame_r.to_ndarray(format="rgb24")  # [H, W, 3]
            break
        container_r.close()

        if decoded is None:
            return tensor
        arr = torch.from_numpy(decoded.astype(np.float32) / 255.0).to(tensor.device, tensor.dtype)
        # Re-embed into original tensor shape (may have been cropped by even-rounding)
        out = tensor.clone()
        out[0, :h, :w] = arr
        return out
    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


def _compress_image_frames(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    if tensor.shape[0] <= 1:
        return _compress_image(tensor, crf)
    return torch.cat([
        _compress_image(tensor[i:i + 1], crf)
        for i in range(tensor.shape[0])
    ], dim=0)


def _build_combined_audio(timeline_data_str: str, duration_frames: int, frame_rate: float, normalize_audio: bool = False) -> dict:
    """Parses timeline JSON, loads/trims audio directly from memory using PyAV, 
    and aligns to a global timeline yielding ComfyUI's format.
    Output length explicitly mimics the timeline's duration_frames length."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    if not timeline_data_str:
        return empty_audio

    try:
        data = json.loads(timeline_data_str)
        audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio

    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        if seg.get("audioFolderAlias") and seg.get("audioFile"):
            try:
                with open(resolve_audio_path(seg["audioFolderAlias"], seg["audioFile"]), "rb") as f:
                    buffer = _io.BytesIO(f.read())
            except Exception as exc:
                log.warning("[PromptRelay] Could not load timeline browser audio %s/%s: %s", seg.get("audioFolderAlias"), seg.get("audioFile"), exc)

        if not buffer and seg.get("audioFile"):
            file_path = os.path.join(folder_paths.get_input_directory(), seg["audioFile"])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        
        if not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except:
                pass
                
        if not buffer:
            continue

        try:
            clip_frames = []
            
            # Use PyAV to decode directly from memory buffer
            with av.open(buffer) as container:
                stream = container.streams.audio[0]
                
                # Setup resampler to ensure output is 44.1kHz, Stereo, Float32 Planar
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )
                
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        # to_ndarray() on fltp gives shape (channels, samples)
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))
                
                # Flush the resampler to get any remaining samples
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            # Concatenate all frame blocks along the samples dimension (dim 1)
            waveform = torch.cat(clip_frames, dim=1) # Shape: [2, total_clip_samples]

            # Calculate interactive trim boundaries
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))

            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples

            if start_sample_src < 0: start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]

            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0: continue

            # Extract the correct segment of the audio
            clip_waveform = waveform[:, start_sample_src:end_sample_src] * _clamp_audio_volume(seg.get("volume", 1.0))

            # Position onto the timeline
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            
            if start_sample_dst >= out_waveform.shape[1]:
                continue
                
            end_sample_dst = start_sample_dst + actual_length

            # Clip any trailing overflow so we don't index past the timeline bounds
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
                
            if actual_length <= 0:
                continue

            # Additive composite (allows clips overlapping to sum together naturally)
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    out_waveform = _normalize_audio_waveform(out_waveform, normalize_audio)
    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    # Split prompts but do NOT filter out empty ones yet, so we can detect them
    locals_list = [p.strip() for p in local_prompts.split("|")]
    
    # Check if any specific segment is empty
    for p in locals_list:
        if not p:
            raise ValueError("There is a segment on the timeline missing a prompt!")

    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        raise ValueError("At least one local prompt is required.")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


class LTXDirector(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirector",
            display_name="LTX Director",
            category="WhatDreamsCost",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The duration_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("audio_vae", optional=True, tooltip="Optional. Connect an Audio VAE to generate audio latents."),
                io.Latent.Input("optional_latent", optional=True, tooltip="Optional. Connect a latent to override the auto-generated one."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Int.Input(
                    "duration_frames", default=120, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.Float.Input(
                    "duration_seconds", default=5, min=0.1, max=1000.0, step=0.01,
                    tooltip="Total timeline duration in seconds (computed/synced from frames).",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.Boolean.Input(
                    "use_custom_audio", default=False, optional=True,
                    tooltip="Toggle between using timeline audio (ON) and generating audio from scratch (OFF).",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "frame_rate", default=24, min=1, max=240, step=1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "display_mode", options=["frames", "seconds"], default="seconds", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
                io.String.Input(
                    "guide_strength", default="",
                    tooltip="Auto-populated from the timeline editor (comma-separated guide strengths for image segments).",
                ),
                io.Boolean.Input(
                    "use_input_image_size", default=False, optional=True,
                    tooltip="Use the first input image size instead of the LTX preset resolution.",
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "4:3", "3:2", "1:1"],
                    default="16:9",
                    optional=True,
                    tooltip="LTX target aspect ratio.",
                ),
                io.Combo.Input(
                    "orientation",
                    options=["landscape", "portrait"],
                    default="landscape",
                    optional=True,
                    tooltip="LTX target orientation.",
                ),
                io.Combo.Input(
                    "quality_tier",
                    options=LTX_QUALITY_TIERS,
                    default="6 - LTX 2.3 native",
                    optional=True,
                    tooltip="LTX resolution quality tier.",
                ),
                io.Combo.Input(
                    "resize_method",
                    options=["maintain aspect ratio", "stretch to fit", "pad", "crop"],
                    default="maintain aspect ratio",
                    optional=True,
                    tooltip="How to resize image segments to fit the target dimensions.",
                ),
                io.Int.Input(
                    "divisible_by", default=32, min=1, max=256, step=1, optional=True,
                    tooltip="Snap the final output image dimensions to be divisible by this number (e.g. 32 for LTX).",
                ),
                io.Int.Input(
                    "img_compression", default=18, min=0, max=100, step=1, optional=True,
                    tooltip="H.264 CRF compression to apply to each guide image. 0 = no compression, higher = more artefacts.",
                ),
                io.Boolean.Input(
                    "use_global_prompt", default=False, optional=True,
                    tooltip="Show the global prompt widget on the node.",
                ),
                io.Boolean.Input(
                    "privacy_mode", default=False, optional=True,
                    tooltip="Encrypt workflow-saved LTX Director state using a local privacy key.",
                ),
                io.String.Input(
                    "privacy_payload", default="", optional=True,
                    tooltip="Encrypted LTX Director state (auto-managed; do not edit by hand).",
                ),
                io.Boolean.Input(
                    "normalize_audio", default=False, optional=True,
                    tooltip="Normalize the final mixed timeline audio to balanced loudness while keeping peaks below a safe ceiling.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="video_latent", tooltip="Auto-generated LTXV empty latent (only populated when no latent is connected)."),
                io.Latent.Output(display_name="audio_latent", tooltip="Auto-generated audio latent (uses custom audio if enabled)."),
                GuideData.Output(display_name="guide_data"),
                io.Float.Output(display_name="frame_rate", tooltip="The frame rate used for the timeline."),
                io.Audio.Output(display_name="combined_audio", tooltip="Combined timeline audio layout."),
                io.Image.Output(display_name="source_video_images", tooltip="Full source video decoded as image frames and resized for stitching."),
                io.Audio.Output(display_name="source_video_audio", tooltip="Audio decoded from the source video, or silence when absent."),
                io.Float.Output(display_name="source_video_frame_rate", tooltip="Original frame rate decoded from the source video."),
                io.Int.Output(display_name="source_video_frame_count", tooltip="Number of frames decoded from the source video."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, global_prompt, duration_frames, duration_seconds,
                timeline_data, local_prompts, segment_lengths, guide_strength="", epsilon=1e-3,
                frame_rate=24, display_mode="seconds",
                use_input_image_size=False, aspect_ratio="16:9", orientation="landscape",
                quality_tier="6 - LTX 2.3 native", resize_method="maintain aspect ratio",
                divisible_by=32, img_compression=0, audio_vae=None, optional_latent=None,
                use_custom_audio=False, use_global_prompt=False,
                privacy_mode=False, privacy_payload="", normalize_audio=False) -> io.NodeOutput:

        frame_rate = _safe_float(frame_rate, 24.0)
        clean_pixel_frames = int(duration_frames) + 1
        clean_latent_frames = ((clean_pixel_frames - 1) // 8) + 1

        resolved_inputs = resolve_ltx_director_inputs(
            global_prompt=global_prompt,
            timeline_data=timeline_data,
            local_prompts=local_prompts,
            segment_lengths=segment_lengths,
            guide_strength=guide_strength,
            duration_frames=duration_frames,
            privacy_mode=privacy_mode,
            privacy_payload=privacy_payload,
        )
        global_prompt = resolved_inputs["global_prompt"]
        timeline_data = resolved_inputs["timeline_data"]
        local_prompts = resolved_inputs["local_prompts"]
        segment_lengths = resolved_inputs["segment_lengths"]
        guide_strength = resolved_inputs["guide_strength"]
        local_prompts = strip_reference_tags_from_prompt_list(local_prompts)

        # --- Build guide_data from image segments FIRST (to derive output dimensions) ---
        guide_data = {
            "images": [],
            "insert_frames": [],
            "strengths": [],
            "frame_rate": frame_rate,
            "reference_images": [],
            "reference_mode": "hidden_tail",
            "clean_pixel_frames": clean_pixel_frames,
            "clean_latent_frames": clean_latent_frames,
            "hidden_reference_count": 0,
        }
        target_w, target_h = _ltx_preset_dimensions(aspect_ratio, orientation, quality_tier)
        derived_w, derived_h = (0, 0) if use_input_image_size else (target_w, target_h)
        source_video_seg = None
        hidden_reference_count = 0
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            reference_usage = build_segment_reference_usage(tdata, duration_frames)
            reference_errors = reference_usage_errors(reference_usage)
            unsupported_tags = reference_errors["unsupported"]
            unknown_tags = reference_errors["unknown"]
            if unsupported_tags:
                log.warning("[PromptRelay] Unsupported reference tags ignored: %s", ", ".join(unsupported_tags))
            if unknown_tags:
                raise LTXDirectorReferenceError(
                    "LTX Director reference tag(s) were used without matching enabled character reference images: "
                    f"{', '.join(unknown_tags)}. "
                    "Add the reference image in the Director References panel or remove the tag from the prompt."
                )
            source_video_seg = next(
                (
                    s for s in tdata.get("segments", [])
                    if s.get("type") == "source_video" and s.get("videoFile")
                ),
                None,
            )
            img_segs = [
                s for s in tdata.get("segments", [])
                if (
                    (
                        s.get("type", "image") == "image"
                        and (s.get("imageFile") or s.get("imageB64"))
                    )
                    or (
                        s.get("type") == "source_video"
                        and s.get("videoFile")
                    )
                )
                and int(s.get("start", 0)) < duration_frames  # exclude segments fully outside duration
            ]
            img_segs.sort(key=lambda s: s["start"])

            strengths = []
            if guide_strength.strip():
                strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]

            configured_references = normalize_reference_images(tdata.get("referenceImages", []))
            timeline_identity_tensors = []
            timeline_identity_insert_frames = []
            timeline_identity_segment_ids = []

            def process_guide_tensor(tensor):
                src_h, src_w = tensor.shape[1], tensor.shape[2]
                if use_input_image_size:
                    return _resize_image_frames(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)
                return _resize_image_frames(tensor, target_w, target_h, resize_method, divisible_by)

            for idx, seg in enumerate(img_segs):
                if seg.get("type") == "source_video":
                    tensor = _load_video_tail_tensor(seg, seg.get("sourceVideoGuideFrames", seg.get("length", 9)))
                else:
                    tensor = _load_image_tensor(seg)

                tensor = process_guide_tensor(tensor)

                # Apply compression
                if img_compression > 0:
                    tensor = _compress_image_frames(tensor, img_compression)

                # In preset mode, keep the generated latent at the selected
                # preset size. Only input-size mode derives dimensions from
                # the processed first guide image.
                if idx == 0 and use_input_image_size:
                    derived_h = tensor.shape[1]
                    derived_w = tensor.shape[2]

                strength = strengths[idx] if idx < len(strengths) else 1.0
                guide_data["images"].append(tensor)
                guide_data["insert_frames"].append(int(seg["start"]))
                guide_data["strengths"].append(float(strength))
                if seg.get("type", "image") == "image":
                    timeline_identity_tensors.append(tensor)
                    timeline_identity_insert_frames.append(int(seg["start"]))
                    timeline_identity_segment_ids.append(seg.get("id"))

            if not configured_references and timeline_identity_tensors:
                fallback_image = _stack_timeline_identity_images(
                    timeline_identity_tensors,
                    target_w,
                    target_h,
                    derived_w,
                    derived_h,
                    divisible_by,
                )
                if fallback_image is not None:
                    guide_data["reference_images"].append(
                        {
                            "id": "timeline-images",
                            "label": "image1",
                            "kind": "timeline_image",
                            "segment_id": ",".join(
                                str(segment_id)
                                for segment_id in timeline_identity_segment_ids
                                if segment_id is not None
                            ),
                            "insert_frame": timeline_identity_insert_frames[0] if timeline_identity_insert_frames else 0,
                            "strength": 1.0,
                            "image": fallback_image,
                        }
                    )

            reference_specs = _dedupe_reference_specs(build_reference_guide_specs(tdata, duration_frames))
            hidden_reference_count = len(reference_specs)
            guide_data["hidden_reference_count"] = hidden_reference_count
            if reference_specs:
                log.warning(
                    "[PromptRelay] Director character references are inserted as hidden tail guide frames "
                    "for likeness. Crop generated latents with LTX Director Crop Reference Tail before decode."
                )
            for spec in reference_specs:
                raw_tensor = _load_image_tensor(_reference_image_segment(spec))
                identity_tensor = _resize_reference_image_frames(
                    raw_tensor,
                    target_w,
                    target_h,
                    derived_w,
                    derived_h,
                    use_input_image_size,
                    divisible_by,
                )
                guide_tensor = _resize_reference_guide_frames(
                    raw_tensor,
                    target_w,
                    target_h,
                    derived_w,
                    derived_h,
                    use_input_image_size,
                    divisible_by,
                )
                if img_compression > 0:
                    identity_tensor = _compress_image_frames(identity_tensor, img_compression)
                    guide_tensor = _compress_image_frames(guide_tensor, img_compression)

                strength = float(spec.get("strength", 1.0))
                hidden_index = len([ref for ref in guide_data["reference_images"] if ref.get("hidden_tail")])
                insert_frame = (clean_latent_frames + hidden_index) * 8
                metadata = {
                    "id": spec.get("id"),
                    "label": spec.get("label"),
                    "kind": spec.get("kind", "character"),
                    "segment_id": spec.get("segment_id"),
                    "insert_frame": insert_frame,
                    "strength": strength,
                    "image": identity_tensor,
                    "hidden_tail": True,
                    "clean_latent_frames": clean_latent_frames,
                    "clean_pixel_frames": clean_pixel_frames,
                }
                guide_data["reference_images"].append(metadata)
                guide_data["images"].append(guide_tensor)
                guide_data["insert_frames"].append(insert_frame)
                guide_data["strengths"].append(strength)

            if guide_data["images"] and (derived_w <= 0 or derived_h <= 0):
                derived_w = target_w
                derived_h = target_h
            
            # If no images were loaded from the timeline, create a dummy image at strength 0
            # to prevent artifacts in text-to-video mode.
            if not guide_data["images"]:
                w = derived_w if derived_w > 0 else 768
                h = derived_h if derived_h > 0 else 512
                w = (w // 32) * 32
                h = (h // 32) * 32
                
                dummy_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
                guide_data["images"].append(dummy_image)
                guide_data["insert_frames"].append(0)
                guide_data["strengths"].append(0.0)
                
                derived_w = w
                derived_h = h
        except LTXDirectorReferenceError:
            raise
        except Exception as e:
            log.warning("[PromptRelay] Could not build guide_data: %s", e)

        # --- Auto-generate LTXV latent if none was provided ---
        total_latents = clean_latent_frames + hidden_reference_count
        ltxv_length = ((total_latents - 1) * 8) + 1
        if optional_latent is None:
            latent_w = max(32, (derived_w // 32) * 32)
            latent_h = max(32, (derived_h // 32) * 32)
            samples = torch.zeros(
                [1, 128, total_latents, latent_h // 32, latent_w // 32],
                device=comfy.model_management.intermediate_device(),
            )
            latent = {"samples": samples}
            log.info(
                "[PromptRelay] Auto-generated LTXV latent: %dx%d, %d pixel frames (%d latent frames, %d hidden refs)",
                latent_w, latent_h, ltxv_length, total_latents, hidden_reference_count,
            )
        else:
            latent = _pad_latent_tail(optional_latent, hidden_reference_count)

        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )

        # --- Build Audio Output ---
        audio_out = _build_combined_audio(timeline_data, ltxv_length, frame_rate, _safe_bool(normalize_audio))
        source_output_w = derived_w if derived_w > 0 else target_w
        source_output_h = derived_h if derived_h > 0 else target_h
        source_resize_method = "maintain aspect ratio" if use_input_image_size else resize_method
        source_video_images, source_video_audio, source_video_frame_rate, source_video_frame_count = _load_source_video_outputs(
            source_video_seg,
            source_output_w,
            source_output_h,
            source_resize_method,
            divisible_by,
            frame_rate,
        )

        # --- Audio Latent Generation ---
        audio_latent = {}
        
        if audio_vae is not None:
            # Helper to generate empty latent
            def get_empty_latent():
                inner = getattr(audio_vae, "first_stage_model", audio_vae)
                z_channels = audio_vae.latent_channels
                audio_freq = inner.latent_frequency_bins
                num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, frame_rate)
                audio_latents = torch.zeros(
                    (1, z_channels, num_audio_latents, audio_freq),
                    device=comfy.model_management.intermediate_device(),
                )
                return {"samples": audio_latents, "type": "audio"}

            if use_custom_audio:
                try:
                    if audio_out is not None:
                        # 1. Encode audio waveform into latent space
                        waveform = audio_out["waveform"]
                        if waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        if waveform.ndim != 3:
                            raise ValueError(
                                f"Expected custom audio waveform with 2 or 3 dims, got shape {tuple(waveform.shape)}"
                            )

                        if hasattr(audio_vae, "first_stage_model"):
                            # ComfyUI's VAE wrapper expects (batch, samples, channels).
                            latent_samples = audio_vae.encode(waveform.movedim(1, -1))
                        else:
                            # Raw LTX AudioVAE expects (batch, channels, samples).
                            latent_samples = audio_vae.encode(
                                waveform,
                                sample_rate=audio_out.get("sample_rate", 44100),
                            )
                        
                        if latent_samples.numel() == 0:
                            raise ValueError("Encoded audio latent is empty (0 elements).")
                        
                        # 2. Create solid mask with value 0.0 (0 means keep/use conditioning, 1 means generate noise)
                        mask = torch.full(
                            (1, latent_samples.shape[-2], latent_samples.shape[-1]), 
                            0.0, 
                            dtype=torch.float32, 
                            device=comfy.model_management.intermediate_device()
                        )
                        
                        # 3. Set Latent Noise Mask
                        audio_latent = {
                            "samples": latent_samples,
                            "type": "audio",
                            "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
                        }
                        log.info("[PromptRelay] Generated custom audio latent with noise mask (value=0.0).")
                    else:
                        raise ValueError("No audio waveform to encode.")
                except Exception as e:
                    log.error("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                # Generate empty latent
                try:
                    audio_latent = get_empty_latent()
                    log.info("[PromptRelay] Auto-generated empty audio latent.")
                except Exception as e:
                    log.error("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

        return io.NodeOutput(
            patched,
            conditioning,
            latent,
            audio_latent,
            guide_data,
            frame_rate,
            audio_out,
            source_video_images,
            source_video_audio,
            float(source_video_frame_rate),
            int(source_video_frame_count),
        )


class LTXDirectorCropReferenceTail(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorCropReferenceTail",
            display_name="LTX Director Crop Reference Tail",
            category="WhatDreamsCost",
            description=(
                "Crops hidden character reference tail frames from a sampled LTX Director latent. "
                "Connect the latent after sampling and guide_data from LTX Director."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Sampled latent to crop back to the visible Director duration."),
                GuideData.Input("guide_data", tooltip="Guide data produced by LTX Director."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="Latent cropped to the visible Director duration."),
                io.Int.Output(display_name="clean_pixel_frames", tooltip="Visible pixel-frame count for downstream video output."),
            ],
        )

    @classmethod
    def execute(cls, latent, guide_data) -> io.NodeOutput:
        clean_latent_frames = None
        clean_pixel_frames = 0
        hidden_reference_count = 0
        if isinstance(guide_data, dict):
            clean_latent_frames = guide_data.get("clean_latent_frames")
            try:
                hidden_reference_count = int(guide_data.get("hidden_reference_count") or 0)
            except (TypeError, ValueError):
                hidden_reference_count = 0
            try:
                clean_pixel_frames = int(guide_data.get("clean_pixel_frames") or 0)
            except (TypeError, ValueError):
                clean_pixel_frames = 0

        if clean_latent_frames is None:
            log.warning(
                "[PromptRelay] LTX Director Crop Reference Tail could not find clean_latent_frames "
                "in guide_data; latent was left unchanged."
            )
            return io.NodeOutput(latent, clean_pixel_frames)

        return io.NodeOutput(
            _crop_latent_to_frame_count(latent, clean_latent_frames, hidden_reference_count),
            clean_pixel_frames,
        )


NODE_CLASS_MAPPINGS = {
    "LTXDirector": LTXDirector,
    "LTXDirectorCropReferenceTail": LTXDirectorCropReferenceTail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "LTXDirectorCropReferenceTail": "LTX Director Crop Reference Tail",
}
