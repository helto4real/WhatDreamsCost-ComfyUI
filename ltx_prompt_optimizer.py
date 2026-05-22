"""Local prompt optimization helpers for LTX Director."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from PIL import Image, ImageOps

try:
    import folder_paths
except Exception:  # noqa: BLE001 - tests can import this module outside ComfyUI.
    folder_paths = None

try:
    from .timeline_image_config import resolve_image_path
except Exception:  # noqa: BLE001 - direct unit-test imports.
    try:
        from timeline_image_config import resolve_image_path
    except Exception:  # noqa: BLE001
        resolve_image_path = None


QWEN_DEPS = ("transformers", "huggingface_hub", "accelerate", "qwen_vl_utils")
FLORENCE_DEPS = ("transformers", "huggingface_hub", "accelerate", "torchvision")
CONFIG_DIR = Path(__file__).resolve().parent / "config"
SETTINGS_FILE = CONFIG_DIR / "ltx_prompt_optimizer_settings.json"
TIMING_FILE = CONFIG_DIR / "ltx_prompt_optimizer_timing.json"
OPTIMIZER_IMAGE_MAX_SIDE = 768
DEFAULT_OPTIMIZER_PROMPT_TEMPLATE = (
    "You are optimizing a local prompt for LTX Director Prompt Relay. "
    "Generate one {rating} video prompt for segment {segment_index} of {segment_total}. "
    "Use provided images only as motion references, not as caption targets. "
    "Infer pose, action, motion direction, expression changes, camera movement, temporal continuation, "
    "and visible or implied sound cues. "
    "Do not describe static image facts like setting, clothing, lighting, object appearance, composition, "
    "or background unless the user explicitly asks or a tiny actor reference is required for clarity. "
    "Write one concise present-tense LTX segment prompt with literal chronological motion. "
    "Do not output bullets, labels, quotes, markdown, negative prompts, or explanations. "
    "Avoid repeated global context and static visual inventory. "
    "User direction to preserve: {direction}. "
    "{continuity}"
)


@dataclass(frozen=True)
class OptimizerModelSpec:
    alias: str
    repo_id: str
    backend: str
    model_subdir: str
    dependencies: tuple[str, ...] = ()


MODEL_REGISTRY: dict[str, OptimizerModelSpec] = {
    "qwen3_vl_8b_quality": OptimizerModelSpec(
        "qwen3_vl_8b_quality",
        "Qwen/Qwen3-VL-8B-Instruct",
        "qwen",
        "VLM",
        QWEN_DEPS,
    ),
    "qwen3_vl_4b_fast": OptimizerModelSpec(
        "qwen3_vl_4b_fast",
        "Qwen/Qwen3-VL-4B-Instruct",
        "qwen",
        "VLM",
        QWEN_DEPS,
    ),
    "qwen3_vl_4b_unredacted": OptimizerModelSpec(
        "qwen3_vl_4b_unredacted",
        "prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1",
        "qwen",
        "VLM",
        QWEN_DEPS,
    ),
    "qwen3_vl_8b_nsfw_caption": OptimizerModelSpec(
        "qwen3_vl_8b_nsfw_caption",
        "monkeyslikebananas/Qwen3-VL-8B-NSFW-Caption-V4.5",
        "qwen",
        "VLM",
        QWEN_DEPS,
    ),
    "qwen2_5_vl_7b_abliterated_legacy": OptimizerModelSpec(
        "qwen2_5_vl_7b_abliterated_legacy",
        "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
        "qwen",
        "VLM",
        QWEN_DEPS,
    ),
    "florence2_fast_caption": OptimizerModelSpec(
        "florence2_fast_caption",
        "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
        "florence",
        "LLM",
        FLORENCE_DEPS,
    ),
    "fallback_text_backend": OptimizerModelSpec(
        "fallback_text_backend",
        "local/fallback-text-backend",
        "fallback",
        "",
        (),
    ),
}

_LOADED_MODELS: dict[str, dict[str, Any]] = {}
_OPTIMIZER_JOBS: dict[str, dict[str, Any]] = {}
_OPTIMIZER_JOBS_LOCK = threading.Lock()
_TIMING_LOCK = threading.Lock()
CUT_SCENE_RE = re.compile(r"\b(cut scene|hard cut|scene cut|new scene|transition)\b", re.I)


class PromptOptimizerError(RuntimeError):
    """Readable optimizer error surfaced through the UI."""


def _noop_status(_message: str, _current: int | None = None, _total: int | None = None) -> None:
    return None


def _progress(
    current: int | None = None,
    total: int | None = None,
    phase: str = "idle",
    percent: float | None = None,
    eta_seconds: float | None = None,
    elapsed_seconds: float | None = None,
    prompt_elapsed_seconds: float | None = None,
    estimated: bool = False,
) -> dict[str, Any]:
    return {
        "current": current,
        "total": total,
        "phase": phase,
        "percent": percent,
        "eta_seconds": eta_seconds,
        "elapsed_seconds": elapsed_seconds,
        "prompt_elapsed_seconds": prompt_elapsed_seconds,
        "estimated": estimated,
    }


def settings_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    return Path(base_dir) / SETTINGS_FILE.name if base_dir is not None else SETTINGS_FILE


def timing_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    return Path(base_dir) / TIMING_FILE.name if base_dir is not None else TIMING_FILE


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    tmp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_optimizer_settings(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = settings_path(base_dir)
    if not path.exists():
        return {"version": 1, "hf_token": "", "prompt_template": ""}
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {"version": 1, "hf_token": "", "prompt_template": ""}
    return {
        "version": 1,
        "hf_token": str(payload.get("hf_token") or ""),
        "prompt_template": str(payload.get("prompt_template") or ""),
    }


def _save_optimizer_settings(settings: dict[str, Any], base_dir: str | os.PathLike[str] | None = None) -> None:
    payload = {
        "version": 1,
        "hf_token": str(settings.get("hf_token") or ""),
        "prompt_template": str(settings.get("prompt_template") or ""),
    }
    if not payload["hf_token"] and not payload["prompt_template"]:
        settings_path(base_dir).unlink(missing_ok=True)
        return
    _write_private_json(settings_path(base_dir), payload)


def save_hf_token(token: str, base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    token = str(token or "").strip()
    settings = load_optimizer_settings(base_dir)
    settings["hf_token"] = token
    if not token:
        clear_hf_token(base_dir)
        return get_optimizer_settings_status(base_dir)
    _save_optimizer_settings(settings, base_dir)
    return get_optimizer_settings_status(base_dir)


def clear_hf_token(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    settings = load_optimizer_settings(base_dir)
    settings["hf_token"] = ""
    _save_optimizer_settings(settings, base_dir)
    return get_optimizer_settings_status(base_dir)


def configured_prompt_template(base_dir: str | os.PathLike[str] | None = None) -> str:
    return str(load_optimizer_settings(base_dir).get("prompt_template") or "").strip()


def active_prompt_template(base_dir: str | os.PathLike[str] | None = None) -> str:
    return configured_prompt_template(base_dir) or DEFAULT_OPTIMIZER_PROMPT_TEMPLATE


def save_prompt_template(template: str, base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    template = str(template or "").strip()
    settings = load_optimizer_settings(base_dir)
    settings["prompt_template"] = template
    _save_optimizer_settings(settings, base_dir)
    return get_optimizer_settings_status(base_dir)


def reset_prompt_template(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    settings = load_optimizer_settings(base_dir)
    settings["prompt_template"] = ""
    _save_optimizer_settings(settings, base_dir)
    return get_optimizer_settings_status(base_dir)


def env_hf_token() -> str:
    return str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()


def configured_hf_token(base_dir: str | os.PathLike[str] | None = None) -> str:
    return str(load_optimizer_settings(base_dir).get("hf_token") or "").strip()


def hf_auth_token(base_dir: str | os.PathLike[str] | None = None) -> str | None:
    return configured_hf_token(base_dir) or env_hf_token() or None


def get_optimizer_settings_status(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    configured = bool(configured_hf_token(base_dir))
    env_available = bool(env_hf_token())
    if configured:
        auth_source = "configured"
    elif env_available:
        auth_source = "environment"
    else:
        auth_source = "anonymous"
    return {
        "ok": True,
        "configPath": str(settings_path(base_dir)),
        "tokenConfigured": configured,
        "envTokenAvailable": env_available,
        "authSource": auth_source,
        "promptTemplate": active_prompt_template(base_dir),
        "defaultPromptTemplate": DEFAULT_OPTIMIZER_PROMPT_TEMPLATE,
        "promptTemplateConfigured": bool(configured_prompt_template(base_dir)),
    }


def model_timing_key(spec: OptimizerModelSpec) -> str:
    return f"{spec.alias}:{spec.backend}"


def load_optimizer_timing(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = timing_path(base_dir)
    if not path.exists():
        return {"version": 1, "profiles": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {"version": 1, "profiles": {}}
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    clean_profiles = {}
    for key, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        try:
            average = float(profile.get("average_seconds") or 0)
            count = int(profile.get("sample_count") or 0)
            last = float(profile.get("last_seconds") or 0)
            updated = float(profile.get("updated_at") or 0)
        except (TypeError, ValueError):
            continue
        if average <= 0 or count <= 0:
            continue
        clean_profiles[str(key)] = {
            "average_seconds": average,
            "sample_count": count,
            "last_seconds": max(0.0, last),
            "updated_at": max(0.0, updated),
        }
    return {"version": 1, "profiles": clean_profiles}


def timing_profile_average(model_key: str, base_dir: str | os.PathLike[str] | None = None) -> float | None:
    profile = load_optimizer_timing(base_dir).get("profiles", {}).get(model_key)
    if not isinstance(profile, dict):
        return None
    average = float(profile.get("average_seconds") or 0)
    return average if average > 0 else None


def record_prompt_timing(
    spec: OptimizerModelSpec,
    duration_seconds: float,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    duration = max(0.001, float(duration_seconds or 0))
    key = model_timing_key(spec)
    with _TIMING_LOCK:
        payload = load_optimizer_timing(base_dir)
        profiles = payload.setdefault("profiles", {})
        previous = profiles.get(key) if isinstance(profiles.get(key), dict) else {}
        count = int(previous.get("sample_count") or 0)
        average = float(previous.get("average_seconds") or 0)
        new_count = count + 1
        new_average = duration if count <= 0 or average <= 0 else average + ((duration - average) / new_count)
        profiles[key] = {
            "average_seconds": new_average,
            "sample_count": new_count,
            "last_seconds": duration,
            "updated_at": time.time(),
        }
        _write_private_json(timing_path(base_dir), payload)
        return profiles[key]


def _models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return Path.cwd() / "models"


def model_path_for(spec: OptimizerModelSpec) -> Path | None:
    if spec.backend == "fallback":
        return None
    return _models_dir() / spec.model_subdir / spec.repo_id.rsplit("/", 1)[-1]


def missing_dependencies(spec: OptimizerModelSpec) -> list[str]:
    return [name for name in spec.dependencies if importlib.util.find_spec(name) is None]


def resolve_model(alias: str | None) -> OptimizerModelSpec:
    key = alias or "fallback_text_backend"
    if key not in MODEL_REGISTRY:
        raise PromptOptimizerError(f"Unknown prompt optimizer model: {key}")
    return MODEL_REGISTRY[key]


def get_model_statuses() -> dict[str, Any]:
    models = []
    for spec in MODEL_REGISTRY.values():
        path = model_path_for(spec)
        missing = missing_dependencies(spec)
        downloaded = bool(path and path.exists())
        if spec.backend == "fallback":
            status = "ready"
        elif missing:
            status = "missing_dependencies"
        elif downloaded:
            status = "downloaded"
        else:
            status = "not_downloaded"
        models.append(
            {
                "alias": spec.alias,
                "repo_id": spec.repo_id,
                "backend": spec.backend,
                "downloaded": downloaded,
                "local_path": str(path) if path else "",
                "missing_dependencies": missing,
                "status": status,
            }
        )
    return {"ok": True, "models": models}


def unload_optimizer_model(alias: str | None = None) -> dict[str, Any]:
    unloaded = []
    if alias:
        spec = resolve_model(alias)
        keys = [spec.alias]
    else:
        keys = list(_LOADED_MODELS.keys())

    torch_modules = []
    for key in keys:
        loaded = _LOADED_MODELS.pop(key, None)
        if not loaded:
            continue
        unloaded.append(key)
        torch_module = loaded.get("torch")
        if torch_module is not None:
            torch_modules.append(torch_module)
        loaded.clear()

    gc.collect()
    for torch_module in torch_modules:
        _clear_torch_cuda_cache(torch_module)

    return {"ok": True, "unloaded": unloaded}


def _clear_torch_cuda_cache(torch_module: Any) -> list[str]:
    actions = []
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return actions
    try:
        if cuda.is_available():
            cuda.empty_cache()
            actions.append("torch.cuda.empty_cache")
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
                actions.append("torch.cuda.ipc_collect")
    except Exception:
        pass
    return actions


def prompt_optimizer_vram_preflight(status_cb: Any = None) -> dict[str, Any]:
    status = status_cb or _noop_status
    status("Releasing Comfy model cache before loading optimizer model...")
    actions = []

    try:
        import comfy.model_management as model_management  # type: ignore[import-not-found]
    except Exception:
        model_management = None

    if model_management is not None:
        for hook_name in ("unload_all_models", "cleanup_models", "soft_empty_cache"):
            hook = getattr(model_management, hook_name, None)
            if not callable(hook):
                continue
            try:
                hook()
                actions.append(f"comfy.model_management.{hook_name}")
            except Exception:
                pass

    gc.collect()
    actions.append("gc.collect")
    try:
        import torch

        actions.extend(_clear_torch_cuda_cache(torch))
    except Exception:
        pass

    return {"ok": True, "actions": actions}


def ensure_model_downloaded(
    spec: OptimizerModelSpec,
    status_cb: Any = None,
) -> Path | None:
    status = status_cb or _noop_status
    path = model_path_for(spec)
    if path is None:
        return None
    status("Checking optional dependencies...")
    if path.exists():
        status(f"Using cached model at {path}")
        return path
    missing = missing_dependencies(spec)
    if missing:
        raise PromptOptimizerError(
            f"Model '{spec.alias}' requires optional packages: {', '.join(missing)}"
        )
    from huggingface_hub import snapshot_download

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        status(f"Downloading {spec.repo_id} into {path}")
        snapshot_download(repo_id=spec.repo_id, local_dir=str(path), local_dir_use_symlinks=False, token=hf_auth_token())
    except Exception as exc:  # noqa: BLE001 - Hugging Face raises several HTTP wrapper types.
        raise _download_error(spec, exc) from exc
    status(f"Downloaded model into {path}")
    return path


def _download_error(spec: OptimizerModelSpec, exc: Exception) -> PromptOptimizerError:
    raw = str(exc)
    lower = raw.lower()
    authish = any(
        marker in lower
        for marker in (
            "401",
            "403",
            "404",
            "repository not found",
            "gated",
            "private",
            "unauthorized",
            "forbidden",
        )
    )
    if authish:
        status = get_optimizer_settings_status()
        token_hint = (
            "A Hugging Face token is configured."
            if status["authSource"] != "anonymous"
            else "No Hugging Face token is configured."
        )
        return PromptOptimizerError(
            f"Could not download '{spec.repo_id}'. The model may be gated, private, moved, or require accepting "
            f"terms on its Hugging Face page. {token_hint} Add or refresh a token in the optimizer settings, "
            f"accept any model access terms in your browser, then try again. Original error: {raw}"
        )
    return PromptOptimizerError(f"Could not download '{spec.repo_id}': {raw}")


def normalize_optimizer_image(image: Image.Image, max_side: int = OPTIMIZER_IMAGE_MAX_SIDE) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image.copy()
    scale = max_side / float(largest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(Path(path)) as image:
        return normalize_optimizer_image(image)


def decode_image(segment: dict[str, Any]) -> Image.Image | None:
    image_data = str(segment.get("image_data") or segment.get("imageData") or "").strip()
    if image_data.startswith("data:image/"):
        try:
            _, encoded = image_data.split(",", 1)
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                return normalize_optimizer_image(image)
        except Exception as exc:  # noqa: BLE001
            raise PromptOptimizerError(f"Could not decode image data for segment '{segment.get('id', '')}': {exc}") from exc

    folder_alias = segment.get("imageFolderAlias")
    image_file = segment.get("imageFile")
    if folder_alias and image_file and resolve_image_path is not None:
        return _load_rgb_image(resolve_image_path(str(folder_alias), str(image_file)))

    if image_file and folder_paths is not None and hasattr(folder_paths, "get_input_directory"):
        candidate = Path(folder_paths.get_input_directory()) / str(image_file)
        if candidate.exists():
            return _load_rgb_image(candidate)

    return None


def clean_prompt_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^(prompt|caption|description)\s*:\s*", "", text, flags=re.I).strip()
    return text.strip(" \t\r\n\"'")


def _sentence_join(parts: list[str]) -> str:
    out = []
    for part in parts:
        part = clean_prompt_text(part)
        if part and part not in out:
            out.append(part)
    return ". ".join(p.rstrip(".") for p in out if p).strip()


def segment_direction_text(segment: dict[str, Any] | None) -> str:
    if not isinstance(segment, dict):
        return ""
    return clean_prompt_text(segment.get("direction") or segment.get("prompt"))


def segment_requests_cut(segment: dict[str, Any]) -> bool:
    return bool(CUT_SCENE_RE.search(segment_direction_text(segment)))


def fallback_optimize_segment(
    segment: dict[str, Any],
    mode: str,
    index: int,
    total: int,
    previous_prompt: str = "",
    next_prompt: str = "",
) -> str:
    direction = clean_prompt_text(segment.get("direction") or segment.get("prompt"))
    label = "opening" if index == 0 else "closing" if index == total - 1 else "continuing"
    cut = segment_requests_cut(segment)
    if direction:
        core = direction
    elif segment.get("type") == "text":
        core = "A text-driven transition continues the scene with clear subject motion"
    else:
        core = "The visible subject moves naturally with clear action and camera movement"

    tone = (
        "Use explicit adult visual language only for visible adult content"
        if mode == "nsfw"
        else "Keep the description cinematic and non-explicit"
    )
    continuity = ""
    if not cut:
        continuity = _sentence_join(
            [
                f"Continue from: {previous_prompt}" if clean_prompt_text(previous_prompt) else "",
                f"Move toward: {next_prompt}" if clean_prompt_text(next_prompt) else "",
            ]
        )
    return _sentence_join(
        [
            core,
            f"{label.capitalize()} moment in the video timeline, described in present tense",
            "focus on action, expression changes, camera motion, temporal movement, and visible or implied sound cues",
            continuity,
            tone,
        ]
    )


def build_optimizer_instruction(
    segment: dict[str, Any],
    mode: str,
    index: int,
    total: int,
    previous_prompt: str = "",
    next_prompt: str = "",
    template: str | None = None,
) -> str:
    direction = clean_prompt_text(segment.get("direction") or segment.get("prompt"))
    rating = "NSFW/unredacted" if mode == "nsfw" else "SFW"
    cut = segment_requests_cut(segment)
    previous_prompt = "" if cut else clean_prompt_text(previous_prompt)
    next_prompt = "" if cut else clean_prompt_text(next_prompt)
    continuity = (
        "Treat this segment as a new cut; do not bridge motion from adjacent segments."
        if cut
        else (
            f"Previous segment motion context: {previous_prompt or 'none'}. "
            f"Next segment motion hint: {next_prompt or 'none'}."
        )
    )
    values = {
        "mode": mode,
        "rating": rating,
        "segment_index": index + 1,
        "segment_total": total,
        "direction": direction or "none",
        "continuity": continuity,
        "previous_prompt": previous_prompt or "none",
        "next_prompt": next_prompt or "none",
        "cut_instruction": "new cut" if cut else "continue naturally",
    }
    try:
        return (template or active_prompt_template()).format_map(values)
    except (KeyError, ValueError) as exc:
        raise PromptOptimizerError(f"Could not format prompt optimizer template: {exc}") from exc


def _load_qwen_model(spec: OptimizerModelSpec, path: Path, status_cb: Any = None) -> dict[str, Any]:
    status = status_cb or _noop_status
    cache_key = spec.alias
    if cache_key in _LOADED_MODELS:
        status(f"Using loaded Qwen model '{spec.alias}'.")
        return _LOADED_MODELS[cache_key]
    status(f"Loading Qwen model from {path}...")
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration if "Qwen3-VL" in spec.repo_id else None
    except Exception:  # noqa: BLE001
        model_cls = None
    if model_cls is None:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(
        str(path),
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    loaded = {"model": model, "processor": processor, "torch": torch}
    _LOADED_MODELS[cache_key] = loaded
    status(f"Loaded Qwen model '{spec.alias}'.")
    return loaded


def _generate_qwen(
    spec: OptimizerModelSpec,
    path: Path,
    images: list[tuple[str, Image.Image]],
    instruction: str,
    status_cb: Any = None,
    loaded: dict[str, Any] | None = None,
) -> str:
    loaded = loaded or _load_qwen_model(spec, path, status_cb)
    model = loaded["model"]
    processor = loaded["processor"]
    torch = loaded["torch"]
    content: list[dict[str, Any]] = []
    image_values = [image for _, image in images]
    for label, image in images:
        content.append({"type": "text", "text": f"{label} image:"})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": instruction})
    conversation = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=image_values or None, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    model_inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    outputs = model.generate(**model_inputs, max_new_tokens=180, do_sample=False, repetition_penalty=1.05)
    input_len = model_inputs["input_ids"].shape[-1]
    return processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _load_florence_model(spec: OptimizerModelSpec, path: Path, status_cb: Any = None) -> dict[str, Any]:
    status = status_cb or _noop_status
    cache_key = spec.alias
    if cache_key in _LOADED_MODELS:
        status(f"Using loaded Florence model '{spec.alias}'.")
        return _LOADED_MODELS[cache_key]
    status(f"Loading Florence model from {path}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        trust_remote_code=True,
        torch_dtype="auto",
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    loaded = {"model": model, "processor": processor, "torch": torch}
    _LOADED_MODELS[cache_key] = loaded
    status(f"Loaded Florence model '{spec.alias}'.")
    return loaded


def _generate_florence(
    spec: OptimizerModelSpec,
    path: Path,
    image: Image.Image | None,
    instruction: str,
    status_cb: Any = None,
    loaded: dict[str, Any] | None = None,
) -> str:
    if image is None:
        return clean_prompt_text(instruction)
    loaded = loaded or _load_florence_model(spec, path, status_cb)
    model = loaded["model"]
    processor = loaded["processor"]
    torch = loaded["torch"]
    inputs = processor(text=instruction, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=180, do_sample=False)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


def _neighbor_segment(segments: list[Any], index: int, offset: int) -> dict[str, Any] | None:
    neighbor_index = index + offset
    if 0 <= neighbor_index < len(segments) and isinstance(segments[neighbor_index], dict):
        return segments[neighbor_index]
    return None


def _previous_context(
    segments: list[Any],
    index: int,
    generated_by_id: dict[str, str],
) -> str:
    previous = _neighbor_segment(segments, index, -1)
    if not previous:
        return ""
    previous_id = str(previous.get("id") or "")
    return clean_prompt_text(generated_by_id.get(previous_id) or segment_direction_text(previous))


def _next_context(segments: list[Any], index: int) -> str:
    return segment_direction_text(_neighbor_segment(segments, index, 1))


def _qwen_context_images(
    segments: list[Any],
    index: int,
    include_neighbors: bool,
) -> list[tuple[str, Image.Image]]:
    offsets = [0] if not include_neighbors else [-1, 0, 1]
    labels = {-1: "Previous", 0: "Current", 1: "Next"}
    images: list[tuple[str, Image.Image]] = []
    for offset in offsets:
        segment = _neighbor_segment(segments, index, offset)
        if not segment:
            continue
        image = decode_image(segment)
        if image is not None:
            images.append((labels[offset], image))
    return images


def optimize_segments(payload: dict[str, Any], status_cb: Any = None) -> dict[str, Any]:
    status = status_cb or _noop_status
    status("Checking selected model...")
    spec = resolve_model(payload.get("model"))
    mode = str(payload.get("mode") or "sfw").lower()
    if mode not in {"sfw", "nsfw"}:
        raise PromptOptimizerError("mode must be 'sfw' or 'nsfw'")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise PromptOptimizerError("segments must be a list")

    selected = [seg for seg in segments if isinstance(seg, dict) and seg.get("selected", True)]
    if not selected:
        raise PromptOptimizerError("Select at least one segment to optimize.")

    path = ensure_model_downloaded(spec, status)
    total = len(segments)
    selected_total = len(selected)
    generated_count = 0
    results = []
    generated_by_id: dict[str, str] = {}
    prompt_template = active_prompt_template()

    for index, segment in enumerate(segments):
        seg_id = str(segment.get("id") or "")
        if not segment.get("selected", True):
            continue
        generated_count += 1
        cut = segment_requests_cut(segment)
        previous_prompt = "" if cut else _previous_context(segments, index, generated_by_id)
        next_prompt = "" if cut else _next_context(segments, index)
        instruction = build_optimizer_instruction(segment, mode, index, total, previous_prompt, next_prompt, prompt_template)

        if spec.backend == "fallback":
            status(f"Generating fallback prompt {generated_count} of {selected_total}...", generated_count, selected_total)
            optimized = fallback_optimize_segment(segment, mode, index, total, previous_prompt, next_prompt)
            status(f"Completed prompt {generated_count} of {selected_total}.", generated_count, selected_total)
        else:
            status(f"Preparing image context {generated_count} of {selected_total}...", generated_count, selected_total)
            if spec.backend == "qwen":
                images = _qwen_context_images(segments, index, not cut)
                prompt_optimizer_vram_preflight(status)
                loaded = _load_qwen_model(spec, path, status)  # type: ignore[arg-type]
                status(f"Generating prompt {generated_count} of {selected_total}...", generated_count, selected_total)
                optimized = _generate_qwen(spec, path, images, instruction, _noop_status, loaded=loaded)  # type: ignore[arg-type]
            elif spec.backend == "florence":
                image = decode_image(segment)
                if image is not None:
                    prompt_optimizer_vram_preflight(status)
                loaded = _load_florence_model(spec, path, status) if image is not None else None  # type: ignore[arg-type]
                status(f"Generating prompt {generated_count} of {selected_total}...", generated_count, selected_total)
                optimized = _generate_florence(spec, path, image, instruction, _noop_status, loaded=loaded)  # type: ignore[arg-type]
            else:
                raise PromptOptimizerError(f"Unsupported optimizer backend: {spec.backend}")
            status(f"Completed prompt {generated_count} of {selected_total}.", generated_count, selected_total)
            status(f"Cleaning generated prompt {generated_count} of {selected_total}...", generated_count, selected_total)
            optimized = clean_prompt_text(optimized)

        if not optimized:
            optimized = fallback_optimize_segment(segment, mode, index, total, previous_prompt, next_prompt)
        generated_by_id[seg_id] = optimized
        results.append({"id": seg_id, "prompt": optimized})

    status(f"Done. Generated {len(results)} prompt{'s' if len(results) != 1 else ''}.", len(results), selected_total)
    return {
        "ok": True,
        "model": spec.alias,
        "mode": mode,
        "results": results,
    }


def _phase_for_message(message: str) -> str:
    lower = message.lower()
    if lower.startswith("generating prompt") or lower.startswith("generating fallback prompt"):
        return "generating"
    if lower.startswith("completed prompt"):
        return "completed_prompt"
    if lower.startswith("cleaning"):
        return "cleaning"
    if lower.startswith("preparing"):
        return "preparing"
    if lower.startswith("downloading"):
        return "downloading"
    if lower.startswith("loading"):
        return "loading"
    if lower.startswith("done"):
        return "completed"
    if lower.startswith("checking") or lower.startswith("using cached") or lower.startswith("downloaded"):
        return "setup"
    return "running"


def _job_average_seconds(job: dict[str, Any]) -> float | None:
    durations = []
    for value in job.get("prompt_durations") or []:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            durations.append(duration)
    if durations:
        return sum(durations) / len(durations)
    average = float(job.get("profile_average_seconds") or 0)
    return average if average > 0 else None


def _estimated_job_progress(job: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    now = now or time.time()
    progress = dict(job.get("progress") or {})
    current = progress.get("current")
    total = progress.get("total")
    phase = str(progress.get("phase") or "idle")
    elapsed = max(0.0, now - float(job.get("created_at") or now))
    prompt_elapsed = None
    percent = progress.get("percent")
    eta_seconds = progress.get("eta_seconds")
    estimated = bool(progress.get("estimated"))

    if job.get("state") == "completed":
        percent = 100.0
        eta_seconds = 0.0
        estimated = False
        phase = "completed"
    elif isinstance(current, int) and isinstance(total, int) and total > 0:
        average = _job_average_seconds(job)
        completed = max(0, min(total, current - 1))
        if phase == "generating":
            started = float(job.get("prompt_started_at") or now)
            prompt_elapsed = max(0.0, now - started)
            if average:
                prompt_fraction = min(0.92, max(0.02, prompt_elapsed / average))
                eta_seconds = max(0.0, average - prompt_elapsed) + (max(total - current, 0) * average)
            else:
                prompt_fraction = min(0.35, max(0.02, prompt_elapsed / 45.0))
                eta_seconds = None
            percent = ((completed + prompt_fraction) / total) * 100.0
            estimated = True
        elif phase in {"completed_prompt", "cleaning", "completed"}:
            percent = (min(current, total) / total) * 100.0
            eta_seconds = (max(total - current, 0) * average) if average else None
            estimated = False
        else:
            percent = (completed / total) * 100.0
            eta_seconds = ((total - completed) * average) if average else None
            estimated = bool(average)

    progress.update(
        {
            "phase": phase,
            "percent": round(max(0.0, min(100.0, float(percent or 0.0))), 1),
            "eta_seconds": round(float(eta_seconds), 1) if eta_seconds is not None else None,
            "elapsed_seconds": round(elapsed, 1),
            "prompt_elapsed_seconds": round(prompt_elapsed, 1) if prompt_elapsed is not None else None,
            "estimated": estimated,
        }
    )
    return progress


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "job_id": job["job_id"],
        "state": job["state"],
        "message": job["message"],
        "progress": _estimated_job_progress(job),
        "results": job.get("results") or [],
        "error": job.get("error") or "",
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def _set_job_status(job_id: str, message: str, current: int | None = None, total: int | None = None) -> None:
    now = time.time()
    with _OPTIMIZER_JOBS_LOCK:
        job = _OPTIMIZER_JOBS.get(job_id)
        if not job:
            return
        phase = _phase_for_message(message)
        previous_phase = str((job.get("progress") or {}).get("phase") or "")
        if phase == "generating" and (previous_phase != "generating" or job.get("prompt_current") != current):
            job["prompt_started_at"] = now
            job["prompt_current"] = current
        elif phase == "completed_prompt":
            started = job.get("prompt_started_at")
            if started is not None and job.get("prompt_current") == current:
                duration = max(0.001, now - float(started))
                job.setdefault("prompt_durations", []).append(duration)
                spec = job.get("model_spec")
                if isinstance(spec, OptimizerModelSpec):
                    record_prompt_timing(spec, duration)
            job["completed_prompts"] = current
            job["prompt_started_at"] = None
        job["message"] = message
        job["progress"] = _progress(current, total, phase=phase)
        job["updated_at"] = now


def start_optimizer_job(payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    with _OPTIMIZER_JOBS_LOCK:
        _OPTIMIZER_JOBS[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "message": "Queued prompt optimization...",
            "progress": _progress(phase="queued", percent=0.0),
            "results": [],
            "error": "",
            "created_at": now,
            "updated_at": now,
            "prompt_durations": [],
        }

    thread = threading.Thread(target=_run_optimizer_job, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id


def _run_optimizer_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        spec = resolve_model(payload.get("model"))
        profile_average = timing_profile_average(model_timing_key(spec))
    except Exception:
        spec = None
        profile_average = None
    with _OPTIMIZER_JOBS_LOCK:
        job = _OPTIMIZER_JOBS.get(job_id)
        if job:
            job["state"] = "running"
            job["message"] = "Starting prompt optimization..."
            job["progress"] = _progress(phase="setup", percent=0.0)
            job["model_spec"] = spec
            job["model_key"] = model_timing_key(spec) if isinstance(spec, OptimizerModelSpec) else ""
            job["profile_average_seconds"] = profile_average
            job["updated_at"] = time.time()
    try:
        result = optimize_segments(payload, lambda message, current=None, total=None: _set_job_status(job_id, message, current, total))
        with _OPTIMIZER_JOBS_LOCK:
            job = _OPTIMIZER_JOBS.get(job_id)
            if job:
                job["state"] = "completed"
                job["message"] = f"Done. Generated {len(result.get('results') or [])} prompt{'s' if len(result.get('results') or []) != 1 else ''}."
                job["progress"] = _progress(
                    len(result.get("results") or []),
                    len(result.get("results") or []),
                    phase="completed",
                    percent=100.0,
                    eta_seconds=0.0,
                    estimated=False,
                )
                job["results"] = result.get("results") or []
                job["error"] = ""
                job["updated_at"] = time.time()
    except Exception as exc:  # noqa: BLE001 - route polls should see readable errors.
        with _OPTIMIZER_JOBS_LOCK:
            job = _OPTIMIZER_JOBS.get(job_id)
            if job:
                job["state"] = "failed"
                job["message"] = "Prompt optimization failed."
                job["error"] = str(exc)
                progress = dict(job.get("progress") or {})
                progress["phase"] = "failed"
                job["progress"] = progress
                job["updated_at"] = time.time()


def get_optimizer_job_status(job_id: str) -> dict[str, Any]:
    with _OPTIMIZER_JOBS_LOCK:
        job = _OPTIMIZER_JOBS.get(str(job_id or ""))
        if not job:
            raise PromptOptimizerError(f"Unknown optimizer job: {job_id}")
        return _job_snapshot(job)
