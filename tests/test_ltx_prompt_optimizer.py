import base64
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from PIL import Image

import ltx_prompt_optimizer as optimizer


class LTXPromptOptimizerTests(unittest.TestCase):
    def test_resolve_model_known_alias(self):
        spec = optimizer.resolve_model("qwen3_vl_4b_fast")
        self.assertEqual(spec.repo_id, "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(spec.backend, "qwen")

    def test_nsfw_caption_alias_uses_live_repo(self):
        spec = optimizer.resolve_model("qwen3_vl_8b_nsfw_caption")
        self.assertEqual(spec.repo_id, "monkeyslikebananas/Qwen3-VL-8B-NSFW-Caption-V4.5")

    def test_resolve_model_rejects_unknown_alias(self):
        with self.assertRaises(optimizer.PromptOptimizerError):
            optimizer.resolve_model("missing")

    def test_model_status_reports_fallback_ready(self):
        statuses = optimizer.get_model_statuses()
        fallback = next(m for m in statuses["models"] if m["alias"] == "fallback_text_backend")
        self.assertEqual(fallback["status"], "ready")
        self.assertEqual(fallback["missing_dependencies"], [])

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

    def test_env_hf_token_fallback(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env_token"}, clear=True):
            with mock.patch.object(optimizer, "configured_hf_token", return_value=""):
                self.assertEqual(optimizer.hf_auth_token(), "hf_env_token")
                status = optimizer.get_optimizer_settings_status()
        self.assertTrue(status["envTokenAvailable"])
        self.assertEqual(status["authSource"], "environment")

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
        self.assertTrue(any("Downloading" in message and str(model_path) in message for message in messages))

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

    def test_decode_folder_image_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/sample.png"
            Image.new("RGB", (5, 4), (0, 255, 0)).save(path)
            with mock.patch.object(optimizer, "resolve_image_path", return_value=path):
                decoded = optimizer.decode_image({"imageFolderAlias": "input", "imageFile": "sample.png"})
        self.assertEqual(decoded.size, (5, 4))

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

    def test_optimize_qwen_uses_generated_previous_and_next_hint(self):
        calls = []

        def fake_generate(_spec, _path, images, instruction, _status):
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
            with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                    result = optimizer.optimize_segments(
                        {"model": "qwen3_vl_4b_fast", "mode": "sfw", "segments": segments}
                    )

        self.assertEqual([item["prompt"] for item in result["results"]], ["generated-1", "generated-2"])
        self.assertEqual([label for label, _ in calls[1][0]], ["Previous", "Current", "Next"])
        self.assertEqual([image for _, image in calls[1][0]], ["image-a", "image-b", "image-c"])
        self.assertIn("Previous segment motion context: generated-1", calls[1][1])
        self.assertIn("Next segment motion hint: She starts laughing", calls[1][1])

    def test_optimize_qwen_cut_uses_current_image_only(self):
        calls = []

        def fake_generate(_spec, _path, images, instruction, _status):
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
                with mock.patch.object(optimizer, "_generate_qwen", side_effect=fake_generate):
                    optimizer.optimize_segments({"model": "qwen3_vl_4b_fast", "mode": "sfw", "segments": segments})

        self.assertEqual([label for label, _ in calls[0][0]], ["Current"])
        self.assertEqual([image for _, image in calls[0][0]], ["image-b"])
        self.assertIn("new cut", calls[0][1])
        self.assertNotIn("She turns toward the camera", calls[0][1])
        self.assertNotIn("She starts laughing", calls[0][1])

    def test_optimize_florence_uses_current_image_only_with_text_context(self):
        calls = []

        def fake_generate(_spec, _path, image, instruction, _status):
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
            with mock.patch.object(optimizer, "decode_image", side_effect=fake_decode):
                with mock.patch.object(optimizer, "_generate_florence", side_effect=fake_generate):
                    result = optimizer.optimize_segments(
                        {"model": "florence2_fast_caption", "mode": "sfw", "segments": segments}
                    )

        self.assertEqual(result["results"][0]["prompt"], "florence generated")
        self.assertEqual(calls[0][0], "image-b")
        self.assertIn("Previous segment motion context: She turns toward the camera", calls[0][1])
        self.assertIn("Next segment motion hint: She starts laughing", calls[0][1])

    def test_optimizer_job_completes(self):
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
        self.assertEqual(status["state"], "completed")
        self.assertEqual([item["id"] for item in status["results"]], ["a"])

    def test_optimizer_job_stores_errors(self):
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
        self.assertEqual(status["state"], "failed")
        self.assertIn("Select at least one segment", status["error"])


if __name__ == "__main__":
    unittest.main()
