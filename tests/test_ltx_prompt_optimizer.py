import base64
import io
import os
from pathlib import Path
import sys
import struct
import tempfile
import time
import types
import unittest
from unittest import mock

from PIL import Image

import ltx_prompt_optimizer as optimizer


class LTXPromptOptimizerTests(unittest.TestCase):
    def _data_url_image(self, width, height, color=(255, 0, 0)):
        image = Image.new("RGB", (width, height), color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _write_fake_gguf(self, path, architecture="gemma4"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = b"general.architecture"
        value = architecture.encode("utf-8")
        payload = [
            b"GGUF",
            struct.pack("<I", 3),
            struct.pack("<Q", 0),
            struct.pack("<Q", 1),
            struct.pack("<Q", len(key)),
            key,
            struct.pack("<I", 8),
            struct.pack("<Q", len(value)),
            value,
        ]
        path.write_bytes(b"".join(payload))

    def test_resolve_model_known_alias(self):
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        self.assertEqual(spec.repo_id, "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(spec.backend, "qwen")

    def test_nsfw_caption_alias_uses_live_repo(self):
        spec = optimizer.resolve_model("qwen3_vl_8b_nsfw_caption")
        self.assertEqual(spec.repo_id, "monkeyslikebananas/Qwen3-VL-8B-NSFW-Caption-V4.5")

    def test_gemma_aliases_use_exact_file_urls(self):
        fp8 = optimizer.resolve_model("gemma4_e4b_it_fp8_scaled")
        self.assertEqual(fp8.backend, "gemma_safetensors")
        self.assertEqual(fp8.file_urls, (optimizer.GEMMA4_E4B_FP8_URL,))

        gguf = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        self.assertEqual(gguf.backend, "llama_cpp_vision")
        self.assertEqual(
            gguf.file_urls,
            (optimizer.GEMMA4_E4B_UNCENSORED_Q8_GGUF_URL, optimizer.GEMMA4_E4B_UNCENSORED_MMPROJ_URL),
        )

    def test_parse_hf_file_url_extracts_repo_revision_and_filename(self):
        parsed = optimizer.parse_hf_file_url(optimizer.GEMMA4_E4B_FP8_URL)
        self.assertEqual(parsed.repo_id, "Comfy-Org/gemma-4")
        self.assertEqual(parsed.revision, "main")
        self.assertEqual(parsed.filename, "text_encoders/gemma4_e4b_it_fp8_scaled.safetensors")

    def test_resolve_model_rejects_unknown_alias(self):
        with self.assertRaises(optimizer.PromptOptimizerError):
            optimizer.resolve_model("missing")

    def test_model_status_reports_fallback_ready(self):
        statuses = optimizer.get_model_statuses()
        fallback = next(m for m in statuses["models"] if m["alias"] == "fallback_text_backend")
        self.assertEqual(fallback["status"], "ready")
        self.assertEqual(fallback["missing_dependencies"], [])

    def test_gguf_status_requires_main_model_and_mmproj(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            main_path = models_dir / spec.model_subdir / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
            main_path.parent.mkdir(parents=True)
            main_path.write_text("fake", encoding="utf-8")
            with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                with mock.patch.object(optimizer, "missing_dependencies", return_value=[]):
                    statuses = optimizer.get_model_statuses()

        model = next(m for m in statuses["models"] if m["alias"] == spec.alias)
        self.assertEqual(model["status"], "not_downloaded")
        self.assertFalse(model["downloaded"])
        self.assertEqual(len(model["local_files"]), 2)

    def test_unload_optimizer_model_removes_loaded_alias_and_clears_cuda(self):
        class FakeCuda:
            def __init__(self):
                self.emptied = False
                self.collected = False

            def is_available(self):
                return True

            def empty_cache(self):
                self.emptied = True

            def ipc_collect(self):
                self.collected = True

        fake_cuda = FakeCuda()
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)
        optimizer._LOADED_MODELS["qwen3_vl_4b_fast"] = {"torch": fake_torch, "model": object()}
        optimizer._LOADED_MODELS["florence2_fast_caption"] = {"model": object()}
        try:
            result = optimizer.unload_optimizer_model("qwen3_vl_4b_fast")
            self.assertEqual(result["unloaded"], ["qwen3_vl_4b_fast"])
            self.assertNotIn("qwen3_vl_4b_fast", optimizer._LOADED_MODELS)
            self.assertIn("florence2_fast_caption", optimizer._LOADED_MODELS)
            self.assertTrue(fake_cuda.emptied)
            self.assertTrue(fake_cuda.collected)
        finally:
            optimizer._LOADED_MODELS.pop("qwen3_vl_4b_fast", None)
            optimizer._LOADED_MODELS.pop("florence2_fast_caption", None)

    def test_unload_optimizer_model_without_alias_clears_all_loaded_models(self):
        optimizer._LOADED_MODELS["qwen3_vl_4b_fast"] = {"model": object()}
        optimizer._LOADED_MODELS["florence2_fast_caption"] = {"model": object()}
        try:
            result = optimizer.unload_optimizer_model()
            self.assertEqual(set(result["unloaded"]), {"qwen3_vl_4b_fast", "florence2_fast_caption"})
            self.assertEqual(optimizer._LOADED_MODELS, {})
        finally:
            optimizer._LOADED_MODELS.clear()

    def test_vram_preflight_calls_comfy_and_torch_cleanup(self):
        calls = []

        class FakeCuda:
            def is_available(self):
                return True

            def empty_cache(self):
                calls.append("empty_cache")

            def ipc_collect(self):
                calls.append("ipc_collect")

        fake_model_management = types.SimpleNamespace(
            unload_all_models=lambda: calls.append("unload_all_models"),
            cleanup_models=lambda: calls.append("cleanup_models"),
            soft_empty_cache=lambda: calls.append("soft_empty_cache"),
        )
        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        fake_comfy = types.ModuleType("comfy")
        fake_comfy.model_management = fake_model_management

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": fake_comfy,
                "comfy.model_management": fake_model_management,
                "torch": fake_torch,
            },
        ):
            result = optimizer.prompt_optimizer_vram_preflight()

        self.assertEqual(result["ok"], True)
        self.assertIn("comfy.model_management.unload_all_models", result["actions"])
        self.assertIn("comfy.model_management.cleanup_models", result["actions"])
        self.assertIn("comfy.model_management.soft_empty_cache", result["actions"])
        self.assertIn("torch.cuda.empty_cache", result["actions"])
        self.assertIn("torch.cuda.ipc_collect", result["actions"])
        self.assertEqual(calls, ["unload_all_models", "cleanup_models", "soft_empty_cache", "empty_cache", "ipc_collect"])

    def test_vram_preflight_succeeds_without_comfy_or_torch(self):
        messages = []
        with mock.patch.dict(sys.modules, {"comfy": None, "torch": None}):
            result = optimizer.prompt_optimizer_vram_preflight(lambda message, *_: messages.append(message))
        self.assertEqual(result["ok"], True)
        self.assertIn("gc.collect", result["actions"])
        self.assertIn("Releasing Comfy model cache", messages[0])

    def test_settings_status_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = optimizer.get_optimizer_settings_status(tmp)
        self.assertFalse(status["tokenConfigured"])
        self.assertFalse(status["envTokenAvailable"])
        self.assertEqual(status["authSource"], "anonymous")

    def test_save_and_clear_hf_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = optimizer.save_hf_token(" hf_test_token ", tmp)
            self.assertTrue(saved["tokenConfigured"])
            self.assertEqual(optimizer.configured_hf_token(tmp), "hf_test_token")

            cleared = optimizer.clear_hf_token(tmp)
            self.assertFalse(cleared["tokenConfigured"])
            self.assertEqual(optimizer.configured_hf_token(tmp), "")

    def test_save_prompt_template_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = "Custom {rating} prompt for {direction}. {continuity}"
            saved = optimizer.save_prompt_template(custom, tmp)
            self.assertTrue(saved["promptTemplateConfigured"])
            self.assertEqual(saved["promptTemplate"], custom)

            reset = optimizer.reset_prompt_template(tmp)
            self.assertFalse(reset["promptTemplateConfigured"])
            self.assertEqual(reset["promptTemplate"], optimizer.DEFAULT_OPTIMIZER_PROMPT_TEMPLATE)

    def test_reference_caption_prompt_file_loads(self):
        text = optimizer.load_reference_caption_prompt_template()
        self.assertIn("{direction}", text)
        self.assertIn("identity conditioning", text)
        self.assertIn("Do not describe actions", text)

    def test_clearing_hf_token_preserves_prompt_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            optimizer.save_hf_token("hf_test_token", tmp)
            optimizer.save_prompt_template("Custom {direction}", tmp)
            cleared = optimizer.clear_hf_token(tmp)
            self.assertFalse(cleared["tokenConfigured"])
            self.assertTrue(cleared["promptTemplateConfigured"])
            self.assertEqual(cleared["promptTemplate"], "Custom {direction}")

    def test_env_hf_token_fallback(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env_token"}, clear=True):
            with mock.patch.object(optimizer, "configured_hf_token", return_value=""):
                self.assertEqual(optimizer.hf_auth_token(), "hf_env_token")
                status = optimizer.get_optimizer_settings_status()
        self.assertTrue(status["envTokenAvailable"])
        self.assertEqual(status["authSource"], "environment")

    def test_timing_status_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = optimizer.load_optimizer_timing(tmp)
        self.assertEqual(status["profiles"], {})

    def test_record_prompt_timing_updates_profile(self):
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        with tempfile.TemporaryDirectory() as tmp:
            first = optimizer.record_prompt_timing(spec, 10.0, tmp)
            second = optimizer.record_prompt_timing(spec, 20.0, tmp)
            stored = optimizer.load_optimizer_timing(tmp)["profiles"][optimizer.model_timing_key(spec)]
        self.assertEqual(first["sample_count"], 1)
        self.assertEqual(second["sample_count"], 2)
        self.assertAlmostEqual(stored["average_seconds"], 15.0)
        self.assertAlmostEqual(stored["last_seconds"], 20.0)

    def test_missing_dependencies_are_reported(self):
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(optimizer.missing_dependencies(spec), list(optimizer.QWEN_DEPS))

    def test_ensure_model_downloaded_passes_hf_token(self):
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "Qwen3-VL-4B-Instruct"
            calls = {}
            messages = []
            fake_hub = types.ModuleType("huggingface_hub")

            def fake_snapshot_download(**kwargs):
                calls.update(kwargs)
                model_path.mkdir(parents=True)

            fake_hub.snapshot_download = fake_snapshot_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                with mock.patch.object(optimizer, "model_path_for", return_value=model_path):
                    with mock.patch.object(optimizer, "missing_dependencies", return_value=[]):
                        with mock.patch.object(optimizer, "hf_auth_token", return_value="hf_saved"):
                            result = optimizer.ensure_model_downloaded(spec, lambda message, *_: messages.append(message))

        self.assertEqual(result, model_path)
        self.assertEqual(calls["repo_id"], spec.repo_id)
        self.assertEqual(calls["token"], "hf_saved")
        self.assertIn("tqdm_class", calls)
        self.assertTrue(any("Downloading" in message and str(model_path) in message for message in messages))

    def test_download_progress_reporter_emits_byte_percent(self):
        updates = []

        def status(message, current=None, total=None, progress=None):
            updates.append((message, current, total, progress))

        reporter = optimizer.DownloadProgressReporter(status, total_bytes=200)
        bar = reporter.tqdm_class("model.gguf", 1, 2, 100)(total=100)
        bar.update(25)
        bar.update(25)

        self.assertEqual(updates[-1][0], "Downloading model.gguf...")
        self.assertEqual(updates[-1][1:3], (1, 2))
        self.assertEqual(updates[-1][3]["download_current_bytes"], 50)
        self.assertEqual(updates[-1][3]["download_total_bytes"], 200)
        self.assertEqual(updates[-1][3]["percent"], 25.0)

    def test_ensure_exact_safetensors_download_uses_hf_hub_download(self):
        spec = optimizer.resolve_model("gemma4_e4b_it_fp8_scaled")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            calls = []
            fake_hub = types.ModuleType("huggingface_hub")

            def fake_hf_hub_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write_fake_gguf(target)
                return str(target)

            fake_hub.hf_hub_download = fake_hf_hub_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                    with mock.patch.object(optimizer, "missing_dependencies", return_value=[]):
                        result = optimizer.ensure_model_downloaded(spec)

        self.assertEqual(result, models_dir / "text_encoders" / "gemma4_e4b_it_fp8_scaled.safetensors")
        self.assertEqual(calls[0]["repo_id"], "Comfy-Org/gemma-4")
        self.assertEqual(calls[0]["revision"], "main")
        self.assertEqual(calls[0]["filename"], "text_encoders/gemma4_e4b_it_fp8_scaled.safetensors")
        self.assertIn("tqdm_class", calls[0])

    def test_ensure_exact_gguf_download_gets_model_and_mmproj(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            calls = []
            fake_hub = types.ModuleType("huggingface_hub")

            def fake_hf_hub_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write_fake_gguf(target)
                return str(target)

            fake_hub.hf_hub_download = fake_hf_hub_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                    with mock.patch.object(optimizer, "missing_dependencies", return_value=[]):
                        result = optimizer.ensure_model_downloaded(spec)

        self.assertEqual(result, models_dir / spec.model_subdir)
        self.assertEqual([call["repo_id"] for call in calls], [spec.repo_id, spec.repo_id])
        self.assertEqual([call["revision"] for call in calls], ["main", "main"])
        self.assertEqual(
            [call["filename"] for call in calls],
            [
                "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
                "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
            ],
        )
        self.assertTrue(all("tqdm_class" in call for call in calls))

    def test_exact_gguf_download_reports_aggregate_progress_across_files(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        sizes = {
            "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf": 100,
            "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf": 300,
        }
        progress_updates = []

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            fake_hub = types.ModuleType("huggingface_hub")

            def fake_hf_hub_url(repo_id, filename, revision=None):
                return f"https://huggingface.co/{repo_id}/resolve/{revision or 'main'}/{filename}"

            def fake_get_hf_file_metadata(url, token=None):
                filename = url.rsplit("/", 1)[-1]
                return types.SimpleNamespace(size=sizes[filename])

            def fake_hf_hub_download(**kwargs):
                filename = kwargs["filename"]
                bar = kwargs["tqdm_class"](total=sizes[filename], desc=filename)
                bar.update(sizes[filename])
                bar.close()
                target = Path(kwargs["local_dir"]) / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write_fake_gguf(target)
                return str(target)

            def status(_message, _current=None, _total=None, progress=None):
                if progress and progress.get("download_total_bytes"):
                    progress_updates.append(progress)

            fake_hub.hf_hub_url = fake_hf_hub_url
            fake_hub.get_hf_file_metadata = fake_get_hf_file_metadata
            fake_hub.hf_hub_download = fake_hf_hub_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                    with mock.patch.object(optimizer, "missing_dependencies", return_value=[]):
                        optimizer.ensure_model_downloaded(spec, status)

        self.assertTrue(any(update["percent"] == 25.0 for update in progress_updates))
        self.assertEqual(progress_updates[-1]["download_current_bytes"], 400)
        self.assertEqual(progress_updates[-1]["download_total_bytes"], 400)
        self.assertEqual(progress_updates[-1]["percent"], 100.0)

    def test_cached_model_status_does_not_say_downloading(self):
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "Qwen3-VL-4B-Instruct"
            model_path.mkdir()
            messages = []
            with mock.patch.object(optimizer, "model_path_for", return_value=model_path):
                result = optimizer.ensure_model_downloaded(spec, lambda message, *_: messages.append(message))
        self.assertEqual(result, model_path)
        self.assertTrue(any("Using cached model" in message for message in messages))
        self.assertFalse(any("Downloading" in message for message in messages))

    def test_download_auth_error_is_readable(self):
        spec = optimizer.resolve_model("qwen3_vl_8b_nsfw_caption")
        err = optimizer._download_error(spec, RuntimeError("404 Client Error: Repository Not Found"))
        self.assertIsInstance(err, optimizer.PromptOptimizerError)
        self.assertIn("gated, private, moved", str(err))
        self.assertIn(spec.repo_id, str(err))

    def test_decode_data_url_image(self):
        image = Image.new("RGB", (8, 6), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        decoded = optimizer.decode_image({"image_data": data_url})

        self.assertEqual(decoded.size, (8, 6))
        self.assertEqual(decoded.mode, "RGB")

    def test_decode_large_data_url_image_downscales_to_optimizer_max_side(self):
        image = Image.new("RGB", (2048, 1024), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        decoded = optimizer.decode_image({"image_data": data_url})

        self.assertEqual(decoded.size, (optimizer.OPTIMIZER_IMAGE_MAX_SIDE, optimizer.OPTIMIZER_IMAGE_MAX_SIDE // 2))
        self.assertEqual(decoded.mode, "RGB")

    def test_normalize_optimizer_image_preserves_small_image_size(self):
        image = Image.new("RGB", (640, 480), (255, 0, 0))
        normalized = optimizer.normalize_optimizer_image(image)
        self.assertEqual(normalized.size, (640, 480))

    def test_normalize_optimizer_image_preserves_aspect_ratio(self):
        image = Image.new("RGB", (3000, 1000), (255, 0, 0))
        normalized = optimizer.normalize_optimizer_image(image)
        self.assertEqual(normalized.size, (optimizer.OPTIMIZER_IMAGE_MAX_SIDE, 256))

    def test_decode_folder_image_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/sample.png"
            Image.new("RGB", (5, 4), (0, 255, 0)).save(path)
            with mock.patch.object(optimizer, "resolve_image_path", return_value=path):
                decoded = optimizer.decode_image({"imageFolderAlias": "input", "imageFile": "sample.png"})
        self.assertEqual(decoded.size, (5, 4))

    def test_decode_large_folder_image_reference_downscales(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/sample.png"
            Image.new("RGB", (1024, 2048), (0, 255, 0)).save(path)
            with mock.patch.object(optimizer, "resolve_image_path", return_value=path):
                decoded = optimizer.decode_image({"imageFolderAlias": "input", "imageFile": "sample.png"})
        self.assertEqual(decoded.size, (optimizer.OPTIMIZER_IMAGE_MAX_SIDE // 2, optimizer.OPTIMIZER_IMAGE_MAX_SIDE))

    def test_prompt_template_differs_by_mode(self):
        segment = {"id": "a", "prompt": "The woman smiles", "type": "image"}
        sfw = optimizer.build_optimizer_instruction(segment, "sfw", 0, 1)
        nsfw = optimizer.build_optimizer_instruction(segment, "nsfw", 0, 1)
        self.assertIn("SFW", sfw)
        self.assertIn("NSFW/unredacted", nsfw)
        self.assertIn("The woman smiles", sfw)
        self.assertIn("motion references, not as caption targets", sfw)
        self.assertIn("visible or implied sound cues", sfw)
        self.assertIn("Do not describe static image facts", sfw)
        self.assertNotIn("Describe the visible subject, setting", sfw)
        self.assertNotIn("camera motion, lighting", sfw)

    def test_prompt_template_includes_continuity_without_cut(self):
        segment = {"id": "b", "prompt": "The woman smiles", "type": "image"}
        text = optimizer.build_optimizer_instruction(
            segment,
            "sfw",
            1,
            3,
            "She turns toward the camera",
            "She starts laughing",
        )
        self.assertIn("Previous segment motion context: She turns toward the camera", text)
        self.assertIn("Next segment motion hint: She starts laughing", text)

    def test_prompt_template_omits_continuity_for_cut(self):
        segment = {"id": "b", "prompt": "hard cut to the woman smiling", "type": "image"}
        text = optimizer.build_optimizer_instruction(
            segment,
            "sfw",
            1,
            3,
            "She turns toward the camera",
            "She starts laughing",
        )
        self.assertIn("new cut", text)
        self.assertNotIn("She turns toward the camera", text)
        self.assertNotIn("She starts laughing", text)

    def test_custom_prompt_template_formats_context(self):
        segment = {"id": "b", "prompt": "The woman smiles", "type": "image"}
        text = optimizer.build_optimizer_instruction(
            segment,
            "nsfw",
            1,
            3,
            "She turns toward the camera",
            "She starts laughing",
            "Make {rating} segment {segment_index}/{segment_total}: {direction}. Prev={previous_prompt}. Next={next_prompt}. {continuity}",
        )
        self.assertIn("NSFW/unredacted", text)
        self.assertIn("segment 2/3", text)
        self.assertIn("The woman smiles", text)
        self.assertIn("Prev=She turns toward the camera", text)
        self.assertIn("Next=She starts laughing", text)

    def test_text_segment_instruction_includes_text_context(self):
        segment = {"id": "b", "prompt": "A quiet pause between shots", "type": "text"}
        text = optimizer.build_optimizer_instruction(
            segment,
            "sfw",
            1,
            3,
            "She turns toward the camera",
            "She starts laughing",
        )
        self.assertIn("text-only timeline segment", text)
        self.assertIn("no current image", text)
        self.assertIn("A quiet pause between shots", text)

    def test_custom_prompt_template_supports_text_placeholders(self):
        segment = {"id": "b", "prompt": "A quiet pause", "type": "text"}
        text = optimizer.build_optimizer_instruction(
            segment,
            "sfw",
            1,
            3,
            template="{segment_type}|{visual_context}|{text_segment_instruction}",
        )
        self.assertIn("text|", text)
        self.assertIn("no current image", text)
        self.assertIn("text-only timeline segment", text)

    def test_reference_caption_instruction_empty_description_is_identity_only(self):
        text = optimizer.build_reference_caption_instruction({"id": "ref1", "label": "image1", "description": ""})
        self.assertIn("stable visual identity details", text)
        self.assertIn("User description to respect: none", text)
        self.assertIn("Do not describe actions", text)

    def test_reference_caption_instruction_respects_user_description(self):
        text = optimizer.build_reference_caption_instruction(
            {"id": "ref1", "label": "image1", "description": "same woman wearing a blue blazer"}
        )
        self.assertIn("same woman wearing a blue blazer", text)
        self.assertIn("follow the user description", text)
        self.assertIn("preserving the subject's stable likeness cues", text)

    def test_fallback_optimize_uses_direction(self):
        text = optimizer.fallback_optimize_segment(
            {"id": "a", "direction": "The woman smiles", "type": "image"},
            "sfw",
            0,
            2,
        )
        self.assertIn("The woman smiles", text)
        self.assertIn("Opening moment", text)
        self.assertIn("visible or implied sound cues", text)
        self.assertNotIn("lighting", text)

    def test_fallback_optimize_uses_continuity_when_no_cut(self):
        text = optimizer.fallback_optimize_segment(
            {"id": "b", "direction": "The woman smiles", "type": "image"},
            "sfw",
            1,
            3,
            "She turns toward the camera",
            "She starts laughing",
        )
        self.assertIn("Continue from: She turns toward the camera", text)
        self.assertIn("Move toward: She starts laughing", text)

    def test_fallback_optimize_omits_continuity_for_cut(self):
        text = optimizer.fallback_optimize_segment(
            {"id": "b", "direction": "new scene, the woman smiles", "type": "image"},
            "sfw",
            1,
            3,
            "She turns toward the camera",
            "She starts laughing",
        )
        self.assertNotIn("Continue from", text)
        self.assertNotIn("Move toward", text)

    def test_optimize_validates_selected_segments(self):
        with self.assertRaises(optimizer.PromptOptimizerError):
            optimizer.optimize_segments({"model": "fallback_text_backend", "mode": "sfw", "segments": []})

    def test_optimize_fallback_returns_selected_only(self):
        messages = []
        result = optimizer.optimize_segments(
            {
                "model": "fallback_text_backend",
                "mode": "sfw",
                "segments": [
                    {"id": "a", "selected": True, "prompt": "A person turns", "type": "image"},
                    {"id": "b", "selected": False, "prompt": "Do not touch", "type": "text"},
                ],
            },
            lambda message, current=None, total=None: messages.append((message, current, total)),
        )
        self.assertEqual(result["ok"], True)
        self.assertEqual([item["id"] for item in result["results"]], ["a"])
        self.assertIn("A person turns", result["results"][0]["prompt"])
        self.assertTrue(any("Checking selected model" in message for message, _, _ in messages))
        self.assertTrue(any("Generating fallback prompt" in message for message, _, _ in messages))
        self.assertTrue(any("Done. Generated 1 prompt" in message for message, _, _ in messages))

    def test_optimize_fallback_returns_mixed_timeline_and_reference_results(self):
        result = optimizer.optimize_segments(
            {
                "model": "fallback_text_backend",
                "mode": "sfw",
                "segments": [{"id": "a", "selected": True, "prompt": "A person turns", "type": "image"}],
                "references": [
                    {
                        "id": "ref1",
                        "label": "image1",
                        "selected": True,
                        "description": "a blonde woman in a green jacket",
                    }
                ],
            }
        )

        self.assertEqual([item["kind"] for item in result["results"]], ["timeline", "reference"])
        self.assertIn("A person turns", result["results"][0]["prompt"])
        self.assertEqual(result["results"][1]["description"], "a blonde woman in a green jacket")

    def test_optimize_fallback_reference_without_description_uses_generic_identity_caption(self):
        result = optimizer.optimize_segments(
            {
                "model": "fallback_text_backend",
                "mode": "sfw",
                "references": [{"id": "ref1", "label": "image1", "selected": True, "description": ""}],
            }
        )

        self.assertEqual(result["results"][0]["kind"], "reference")
        self.assertIn("image1", result["results"][0]["description"])
        self.assertIn("identity features", result["results"][0]["description"])

    def test_optimize_qwen_uses_generated_previous_and_next_hint(self):
        calls = []

        def fake_generate(_spec, _path, images, instruction, _status, loaded=None):
            calls.append((images, instruction))
            return f"generated-{len(calls)}"

        def fake_decode(segment):
            return f"image-{segment['id']}"

        segments = [
            {"id": "a", "selected": True, "prompt": "She turns toward the camera", "type": "image"},
            {"id": "b", "selected": True, "prompt": "She smiles wider", "type": "image"},
            {"id": "c", "selected": False, "prompt": "She starts laughing", "type": "image"},
        ]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/qwen")):
            with mock.patch.object(optimizer, "active_prompt_template", return_value=optimizer.DEFAULT_OPTIMIZER_PROMPT_TEMPLATE):
                with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                    with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight"):
                        with mock.patch.object(optimizer, "_load_qwen_model", return_value={"loaded": True}):
                            with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                                result = optimizer.optimize_segments(
                                    {"model": "qwen3_vl_4b_fast", "mode": "sfw", "segments": segments}
                                )

        self.assertEqual([item["prompt"] for item in result["results"]], ["generated-1", "generated-2"])
        self.assertEqual([label for label, _ in calls[1][0]], ["Previous", "Current", "Next"])
        self.assertEqual([image for _, image in calls[1][0]], ["image-a", "image-b", "image-c"])
        self.assertIn("Previous segment motion context: generated-1", calls[1][1])
        self.assertIn("Next segment motion hint: She starts laughing", calls[1][1])

    def test_qwen_context_images_are_downscaled(self):
        segments = [
            {"id": "a", "selected": False, "image_data": self._data_url_image(2048, 1024)},
            {"id": "b", "selected": True, "image_data": self._data_url_image(1024, 2048)},
            {"id": "c", "selected": False, "image_data": self._data_url_image(3000, 1000)},
        ]

        images = optimizer._qwen_context_images(segments, 1, True)

        self.assertEqual([label for label, _ in images], ["Previous", "Current", "Next"])
        self.assertEqual([image.size for _, image in images], [(768, 384), (384, 768), (768, 256)])

    def test_qwen_text_context_images_use_previous_and_next_visuals(self):
        def fake_decode(segment):
            return None if segment["id"] == "b" else f"image-{segment['id']}"

        segments = [
            {"id": "a", "prompt": "She turns", "type": "image"},
            {"id": "b", "prompt": "A pause", "type": "text"},
            {"id": "c", "prompt": "She laughs", "type": "image"},
        ]
        with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
            images = optimizer._qwen_context_images(segments, 1, True)
        self.assertEqual([label for label, _ in images], ["Previous", "Next"])
        self.assertEqual([image for _, image in images], ["image-a", "image-c"])

    def test_qwen_text_cut_uses_no_neighbor_images(self):
        segments = [
            {"id": "a", "prompt": "She turns", "type": "image"},
            {"id": "b", "prompt": "hard cut to black", "type": "text"},
            {"id": "c", "prompt": "She laughs", "type": "image"},
        ]
        with mock.patch.object(optimizer, "decode_image", return_value="image"):
            images = optimizer._qwen_context_images(segments, 1, False)
        self.assertEqual(images, [])

    def test_llama_cpp_model_paths_resolve_exact_gguf_and_mmproj(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            model_dir = models_dir / spec.model_subdir
            model_path = model_dir / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
            mmproj_path = model_dir / "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
            self._write_fake_gguf(model_path)
            self._write_fake_gguf(mmproj_path, architecture="clip")
            with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                resolved = optimizer._llama_cpp_model_paths(spec, model_dir)

        self.assertEqual(resolved, (model_path, mmproj_path))

    def test_llama_cpp_model_paths_missing_exact_file_reports_discovered_alternates(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            model_dir = models_dir / spec.model_subdir
            alternate = model_dir / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
            mmproj_path = model_dir / "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
            self._write_fake_gguf(alternate)
            self._write_fake_gguf(mmproj_path, architecture="clip")
            with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                with self.assertRaises(optimizer.PromptOptimizerError) as ctx:
                    optimizer._llama_cpp_model_paths(spec, model_dir)

        message = str(ctx.exception)
        self.assertIn("missing the expected main model GGUF file", message)
        self.assertIn("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", message)
        self.assertIn("mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf", message)

    def test_validate_gguf_file_rejects_invalid_magic_with_redownload_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.gguf"
            path.write_bytes(b"not-a-gguf")
            with self.assertRaises(optimizer.PromptOptimizerError) as ctx:
                optimizer.validate_gguf_file(path, "main model GGUF")

        message = str(ctx.exception)
        self.assertIn("not a valid GGUF file", message)
        self.assertIn("download it again", message)

    def test_load_llama_cpp_vision_uses_mmproj_file(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            model_dir = models_dir / spec.model_subdir
            model_path = model_dir / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
            mmproj_path = model_dir / "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
            self._write_fake_gguf(model_path)
            self._write_fake_gguf(mmproj_path, architecture="clip")
            calls = {}

            class FakeChatHandler:
                def __init__(self, clip_model_path):
                    calls["clip_model_path"] = clip_model_path

            class FakeLlama:
                def __init__(self, **kwargs):
                    calls["llama_kwargs"] = kwargs

            fake_llama_cpp = types.ModuleType("llama_cpp")
            fake_llama_cpp.Llama = FakeLlama
            fake_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
            fake_chat_format.Llava15ChatHandler = FakeChatHandler
            with mock.patch.dict(
                sys.modules,
                {
                    "llama_cpp": fake_llama_cpp,
                    "llama_cpp.llama_chat_format": fake_chat_format,
                },
            ):
                with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                    try:
                        loaded = optimizer._load_llama_cpp_vision_model(spec, model_dir)
                    finally:
                        optimizer._LOADED_MODELS.pop(spec.alias, None)

        self.assertEqual(calls["clip_model_path"], str(mmproj_path))
        self.assertEqual(calls["llama_kwargs"]["model_path"], str(model_path))
        self.assertEqual(loaded["mmproj_path"], mmproj_path)

    def test_load_llama_cpp_vision_gemma4_failure_mentions_runtime_support(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            model_dir = models_dir / spec.model_subdir
            model_path = model_dir / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
            mmproj_path = model_dir / "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
            self._write_fake_gguf(model_path)
            self._write_fake_gguf(mmproj_path, architecture="clip")

            class FakeChatHandler:
                def __init__(self, clip_model_path):
                    self.clip_model_path = clip_model_path

            class FakeLlama:
                def __init__(self, **_kwargs):
                    raise RuntimeError("Failed to load model from file")

            fake_llama_cpp = types.ModuleType("llama_cpp")
            fake_llama_cpp.Llama = FakeLlama
            fake_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
            fake_chat_format.Llava15ChatHandler = FakeChatHandler
            with mock.patch.dict(
                sys.modules,
                {
                    "llama_cpp": fake_llama_cpp,
                    "llama_cpp.llama_chat_format": fake_chat_format,
                },
            ):
                with mock.patch.object(optimizer, "_models_dir", return_value=models_dir):
                    with self.assertRaises(optimizer.PromptOptimizerError) as ctx:
                        optimizer._load_llama_cpp_vision_model(spec, model_dir)

        message = str(ctx.exception)
        self.assertIn("Gemma 4/K_P", message)
        self.assertIn("Upgrade or reinstall llama-cpp-python", message)
        self.assertIn(str(model_path), message)

    def test_generate_llama_cpp_vision_sends_image_data_url(self):
        spec = optimizer.resolve_model("gemma4_e4b_uncensored_gguf_q8")
        calls = {}

        class FakeLlama:
            def create_chat_completion(self, **kwargs):
                calls.update(kwargs)
                return {"choices": [{"message": {"content": "generated llama prompt"}}]}

        image = Image.new("RGB", (4, 3), (255, 0, 0))
        result = optimizer._generate_llama_cpp_vision(
            spec,
            Path("/tmp/model"),
            [("Current", image)],
            "Write a prompt",
            loaded={"model": FakeLlama()},
        )

        self.assertEqual(result, "generated llama prompt")
        content = calls["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Current image:"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[-1], {"type": "text", "text": "Write a prompt"})

    def test_gemma_safetensors_generation_reports_non_generator(self):
        spec = optimizer.resolve_model("gemma4_e4b_it_fp8_scaled")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gemma4_e4b_it_fp8_scaled.safetensors"
            path.write_text("fake", encoding="utf-8")
            with self.assertRaises(optimizer.PromptOptimizerError) as ctx:
                optimizer._generate_gemma_safetensors(spec, path, "Write a prompt")

        self.assertIn("not a standalone prompt-generating optimizer model", str(ctx.exception))

    def test_optimize_qwen_cut_uses_current_image_only(self):
        calls = []

        def fake_generate(_spec, _path, images, instruction, _status, loaded=None):
            calls.append((images, instruction))
            return "generated"

        def fake_decode(segment):
            return f"image-{segment['id']}"

        segments = [
            {"id": "a", "selected": False, "prompt": "She turns toward the camera", "type": "image"},
            {"id": "b", "selected": True, "prompt": "cut scene, she smiles wider", "type": "image"},
            {"id": "c", "selected": False, "prompt": "She starts laughing", "type": "image"},
        ]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/qwen")):
            with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight"):
                    with mock.patch.object(optimizer, "_load_qwen_model", return_value={"loaded": True}):
                        with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                            optimizer.optimize_segments({"model": "qwen3_vl_4b_fast", "mode": "sfw", "segments": segments})

        self.assertEqual([label for label, _ in calls[0][0]], ["Current"])
        self.assertEqual([image for _, image in calls[0][0]], ["image-b"])
        self.assertIn("new cut", calls[0][1])
        self.assertNotIn("She turns toward the camera", calls[0][1])
        self.assertNotIn("She starts laughing", calls[0][1])

    def test_optimize_qwen_text_segment_uses_neighbor_images(self):
        calls = []

        def fake_generate(_spec, _path, images, instruction, _status, loaded=None):
            calls.append((images, instruction))
            return "generated text prompt"

        def fake_decode(segment):
            return None if segment["type"] == "text" else f"image-{segment['id']}"

        segments = [
            {"id": "a", "selected": False, "prompt": "She turns toward the camera", "type": "image"},
            {"id": "b", "selected": True, "prompt": "A quiet pause", "type": "text"},
            {"id": "c", "selected": False, "prompt": "She starts laughing", "type": "image"},
        ]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/qwen")):
            with mock.patch.object(optimizer, "active_prompt_template", return_value=optimizer.DEFAULT_OPTIMIZER_PROMPT_TEMPLATE):
                with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                    with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight"):
                        with mock.patch.object(optimizer, "_load_qwen_model", return_value={"loaded": True}):
                            with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                                result = optimizer.optimize_segments(
                                    {"model": "qwen3_vl_4b_fast", "mode": "sfw", "segments": segments}
                                )

        self.assertEqual(result["results"][0]["prompt"], "generated text prompt")
        self.assertEqual([label for label, _ in calls[0][0]], ["Previous", "Next"])
        self.assertEqual([image for _, image in calls[0][0]], ["image-a", "image-c"])
        self.assertIn("text-only timeline segment", calls[0][1])

    def test_optimize_qwen_keeps_generation_phase_after_loading(self):
        messages = []

        def fake_load(_spec, _path, status):
            status("Using loaded Qwen model 'qwen3_vl_4b_fast'.")
            return {"loaded": True}

        def fake_generate(_spec, _path, _images, _instruction, status, loaded=None):
            status("Internal load status that should be ignored.")
            return "generated"

        segments = [{"id": "a", "selected": True, "prompt": "She smiles wider", "type": "image"}]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/qwen")):
            with mock.patch.object(optimizer, "decode_image", return_value=None):
                with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight"):
                    with mock.patch.object(optimizer, "_load_qwen_model", side_effect=fake_load):
                        with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                            optimizer.optimize_segments(
                                {"model": "qwen3_vl_4b_fast", "mode": "sfw", "segments": segments},
                                lambda message, current=None, total=None: messages.append(message),
                            )

        load_index = messages.index("Using loaded Qwen model 'qwen3_vl_4b_fast'.")
        generate_index = messages.index("Generating prompt 1 of 1...")
        completed_index = messages.index("Completed prompt 1 of 1.")
        self.assertLess(load_index, generate_index)
        self.assertLess(generate_index, completed_index)
        self.assertNotIn("Internal load status that should be ignored.", messages)

    def test_optimize_qwen_runs_vram_preflight_before_model_load(self):
        messages = []
        calls = []

        def fake_preflight(status):
            calls.append("preflight")
            status("Releasing Comfy model cache before loading optimizer model...")

        def fake_load(_spec, _path, _status):
            calls.append("load")
            return {"loaded": True}

        def fake_generate(_spec, _path, _images, _instruction, _status, loaded=None):
            calls.append("generate")
            return "generated"

        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/qwen")):
            with mock.patch.object(optimizer, "decode_image", return_value=None):
                with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight", side_effect=fake_preflight):
                    with mock.patch.object(optimizer, "_load_qwen_model", side_effect=fake_load):
                        with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                            optimizer.optimize_segments(
                                {
                                    "model": "qwen3_vl_4b_fast",
                                    "mode": "sfw",
                                    "segments": [{"id": "a", "selected": True, "prompt": "She smiles", "type": "image"}],
                                },
                                lambda message, current=None, total=None: messages.append(message),
                            )

        self.assertEqual(calls, ["preflight", "load", "generate"])
        self.assertIn("Releasing Comfy model cache before loading optimizer model...", messages)

    def test_optimize_fallback_skips_vram_preflight(self):
        with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight") as preflight:
            optimizer.optimize_segments(
                {
                    "model": "fallback_text_backend",
                    "mode": "sfw",
                    "segments": [{"id": "a", "selected": True, "prompt": "A person turns", "type": "image"}],
                }
            )
        preflight.assert_not_called()

    def test_optimize_florence_uses_current_image_only_with_text_context(self):
        calls = []

        def fake_generate(_spec, _path, image, instruction, _status, loaded=None):
            calls.append((image, instruction))
            return "florence generated"

        def fake_decode(segment):
            return f"image-{segment['id']}"

        segments = [
            {"id": "a", "selected": False, "prompt": "She turns toward the camera", "type": "image"},
            {"id": "b", "selected": True, "prompt": "She smiles wider", "type": "image"},
            {"id": "c", "selected": False, "prompt": "She starts laughing", "type": "image"},
        ]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/florence")):
            with mock.patch.object(optimizer, "active_prompt_template", return_value=optimizer.DEFAULT_OPTIMIZER_PROMPT_TEMPLATE):
                with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                    with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight"):
                        with mock.patch.object(optimizer, "_load_florence_model", return_value={"loaded": True}):
                            with mock.patch.object(optimizer, "_generate_florence", side_effect=fake_generate):
                                result = optimizer.optimize_segments(
                                    {"model": "florence2_fast_caption", "mode": "sfw", "segments": segments}
                                )

        self.assertEqual(result["results"][0]["prompt"], "florence generated")
        self.assertEqual(calls[0][0], "image-b")
        self.assertIn("Previous segment motion context: She turns toward the camera", calls[0][1])
        self.assertIn("Next segment motion hint: She starts laughing", calls[0][1])

    def test_optimize_florence_text_segment_uses_nearest_neighbor_image(self):
        calls = []

        def fake_generate(_spec, _path, image, instruction, _status, loaded=None):
            calls.append((image, instruction))
            return "florence text generated"

        def fake_decode(segment):
            return None if segment["type"] == "text" else f"image-{segment['id']}"

        segments = [
            {"id": "a", "selected": False, "prompt": "She turns toward the camera", "type": "image"},
            {"id": "b", "selected": True, "prompt": "A quiet pause", "type": "text"},
            {"id": "c", "selected": False, "prompt": "She starts laughing", "type": "image"},
        ]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/florence")):
            with mock.patch.object(optimizer, "active_prompt_template", return_value=optimizer.DEFAULT_OPTIMIZER_PROMPT_TEMPLATE):
                with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                    with mock.patch.object(optimizer, "prompt_optimizer_vram_preflight"):
                        with mock.patch.object(optimizer, "_load_florence_model", return_value={"loaded": True}):
                            with mock.patch.object(optimizer, "_generate_florence", side_effect=fake_generate):
                                result = optimizer.optimize_segments(
                                    {"model": "florence2_fast_caption", "mode": "sfw", "segments": segments}
                                )

        self.assertEqual(result["results"][0]["prompt"], "florence text generated")
        self.assertEqual(calls[0][0], "image-a")
        self.assertIn("text-only timeline segment", calls[0][1])

    def test_optimize_florence_text_segment_without_images_uses_fallback(self):
        segments = [{"id": "b", "selected": True, "prompt": "", "type": "text"}]
        with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/florence")):
            with mock.patch.object(optimizer, "decode_image", return_value=None):
                with mock.patch.object(optimizer, "_load_florence_model") as load_model:
                    result = optimizer.optimize_segments(
                        {"model": "florence2_fast_caption", "mode": "sfw", "segments": segments}
                    )

        load_model.assert_not_called()
        self.assertIn("text-driven timeline section", result["results"][0]["prompt"].lower())

    def test_optimizer_job_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(optimizer, "TIMING_FILE", Path(tmp) / "timing.json"):
                job_id = optimizer.start_optimizer_job(
                    {
                        "model": "fallback_text_backend",
                        "mode": "sfw",
                        "segments": [{"id": "a", "selected": True, "prompt": "A person turns", "type": "image"}],
                    }
                )
                status = optimizer.get_optimizer_job_status(job_id)
                deadline = time.time() + 2
                while status["state"] in {"queued", "running"} and time.time() < deadline:
                    time.sleep(0.01)
                    status = optimizer.get_optimizer_job_status(job_id)
                timing = optimizer.load_optimizer_timing()
        self.assertEqual(status["state"], "completed")
        self.assertEqual([item["id"] for item in status["results"]], ["a"])
        self.assertEqual(status["progress"]["percent"], 100.0)
        self.assertEqual(status["progress"]["eta_seconds"], 0.0)
        profile = timing["profiles"][optimizer.model_timing_key(optimizer.resolve_model("fallback_text_backend"))]
        self.assertEqual(profile["sample_count"], 1)

    def test_optimizer_job_stores_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(optimizer, "TIMING_FILE", Path(tmp) / "timing.json"):
                job_id = optimizer.start_optimizer_job(
                    {
                        "model": "fallback_text_backend",
                        "mode": "sfw",
                        "segments": [],
                    }
                )
                status = optimizer.get_optimizer_job_status(job_id)
                deadline = time.time() + 2
                while status["state"] in {"queued", "running"} and time.time() < deadline:
                    time.sleep(0.01)
                    status = optimizer.get_optimizer_job_status(job_id)
                timing = optimizer.load_optimizer_timing()
        self.assertEqual(status["state"], "failed")
        self.assertIn("Select at least one segment", status["error"])
        self.assertEqual(timing["profiles"], {})

    def test_failed_generation_does_not_update_timing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(optimizer, "TIMING_FILE", Path(tmp) / "timing.json"):
                with mock.patch.object(optimizer, "ensure_model_downloaded", return_value=Path("/tmp/qwen")):
                    with mock.patch.object(optimizer, "decode_image", return_value=None):
                        with mock.patch.object(optimizer, "_load_qwen_model", return_value={"loaded": True}):
                            with mock.patch.object(optimizer, "_generate_qwen", side_effect=RuntimeError("boom")):
                                job_id = optimizer.start_optimizer_job(
                                    {
                                        "model": "qwen3_vl_4b_fast",
                                        "mode": "sfw",
                                        "segments": [{"id": "a", "selected": True, "prompt": "A person turns", "type": "image"}],
                                    }
                                )
                                status = optimizer.get_optimizer_job_status(job_id)
                                deadline = time.time() + 2
                                while status["state"] in {"queued", "running"} and time.time() < deadline:
                                    time.sleep(0.01)
                                    status = optimizer.get_optimizer_job_status(job_id)
                timing = optimizer.load_optimizer_timing()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(timing["profiles"], {})

    def test_active_generation_status_estimates_percent_and_eta(self):
        now = time.time()
        job_id = "estimated-progress-test"
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        with optimizer._OPTIMIZER_JOBS_LOCK:
            optimizer._OPTIMIZER_JOBS[job_id] = {
                "job_id": job_id,
                "state": "running",
                "message": "Generating prompt 2 of 4...",
                "progress": optimizer._progress(2, 4, phase="generating"),
                "results": [],
                "error": "",
                "created_at": now - 6,
                "updated_at": now - 1,
                "model_spec": spec,
                "model_key": optimizer.model_timing_key(spec),
                "profile_average_seconds": 10.0,
                "prompt_started_at": now - 5,
                "prompt_current": 2,
                "prompt_durations": [],
            }
        try:
            status = optimizer.get_optimizer_job_status(job_id)
        finally:
            with optimizer._OPTIMIZER_JOBS_LOCK:
                optimizer._OPTIMIZER_JOBS.pop(job_id, None)
        self.assertEqual(status["progress"]["phase"], "generating")
        self.assertTrue(status["progress"]["estimated"])
        self.assertGreater(status["progress"]["percent"], 25.0)
        self.assertLess(status["progress"]["percent"], 50.0)
        self.assertGreater(status["progress"]["eta_seconds"], 0)

    def test_download_status_preserves_reported_percent(self):
        now = time.time()
        job_id = "download-progress-test"
        with optimizer._OPTIMIZER_JOBS_LOCK:
            optimizer._OPTIMIZER_JOBS[job_id] = {
                "job_id": job_id,
                "state": "running",
                "message": "Downloading model.gguf...",
                "progress": optimizer._progress(
                    1,
                    2,
                    phase="downloading",
                    percent=42.5,
                    download_current_bytes=425,
                    download_total_bytes=1000,
                    download_file="model.gguf",
                    download_file_index=1,
                    download_file_total=2,
                ),
                "results": [],
                "error": "",
                "created_at": now - 3,
                "updated_at": now,
                "prompt_durations": [],
            }
        try:
            status = optimizer.get_optimizer_job_status(job_id)
        finally:
            with optimizer._OPTIMIZER_JOBS_LOCK:
                optimizer._OPTIMIZER_JOBS.pop(job_id, None)

        self.assertEqual(status["progress"]["phase"], "downloading")
        self.assertEqual(status["progress"]["percent"], 42.5)
        self.assertEqual(status["progress"]["download_current_bytes"], 425)
        self.assertEqual(status["progress"]["download_total_bytes"], 1000)
        self.assertFalse(status["progress"]["estimated"])


if __name__ == "__main__":
    unittest.main()
