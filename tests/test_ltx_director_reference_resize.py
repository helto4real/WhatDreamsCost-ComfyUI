import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "wdc_ltx_director_resize_test"


def _install_stubs():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(ROOT)
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: "cpu"
    comfy.model_management = model_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management

    class _BaseType:
        class Input:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Output:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

    class _Io:
        ComfyNode = object
        Audio = _BaseType
        Boolean = _BaseType
        Clip = _BaseType
        Combo = _BaseType
        Conditioning = _BaseType
        Float = _BaseType
        Image = _BaseType
        Int = _BaseType
        Latent = _BaseType
        Model = _BaseType
        String = _BaseType

        @staticmethod
        def Custom(_name):
            return type("CustomType", (_BaseType,), {})

        class Schema:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def NodeOutput(*values):
            return values

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _Io
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    prompt_relay = types.ModuleType(f"{PACKAGE}.prompt_relay")
    for name in (
        "get_raw_tokenizer",
        "map_token_indices",
        "build_segments",
        "create_mask_fn",
        "distribute_segment_lengths",
    ):
        setattr(prompt_relay, name, lambda *args, **kwargs: None)
    sys.modules[f"{PACKAGE}.prompt_relay"] = prompt_relay

    patches = types.ModuleType(f"{PACKAGE}.patches")
    patches.detect_model_type = lambda *args, **kwargs: None
    patches.apply_patches = lambda model, *args, **kwargs: model
    sys.modules[f"{PACKAGE}.patches"] = patches

    image_config = types.ModuleType(f"{PACKAGE}.timeline_image_config")
    image_config.resolve_image_path = lambda folder_alias, image_file: image_file
    sys.modules[f"{PACKAGE}.timeline_image_config"] = image_config

    audio_config = types.ModuleType(f"{PACKAGE}.timeline_audio_config")
    audio_config.resolve_audio_path = lambda audio_file: audio_file
    sys.modules[f"{PACKAGE}.timeline_audio_config"] = audio_config

    privacy = types.ModuleType(f"{PACKAGE}.ltx_director_privacy")
    privacy.resolve_ltx_director_inputs = lambda **kwargs: kwargs
    sys.modules[f"{PACKAGE}.ltx_director_privacy"] = privacy


def _load_ltx_director():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.ltx_director", ROOT / "ltx_director.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{PACKAGE}.ltx_director"] = module
    spec.loader.exec_module(module)
    return module


ltx_director = _load_ltx_director()


class FakeNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


class LTXDirectorReferenceResizeTests(unittest.TestCase):
    def _execute_director_for_guide_data(
        self,
        timeline,
        image_map,
        *,
        video_map=None,
        duration_frames=16,
        local_prompts="",
        segment_lengths="16",
        guide_strength="",
        use_input_image_size=False,
        resize_method="maintain aspect ratio",
        return_full_result=False,
        optional_latent=None,
    ):
        video_map = video_map or {}

        def fake_load_image(segment):
            return image_map.get(segment.get("imageFile"), torch.zeros((1, 32, 32, 3), dtype=torch.float32))

        def fake_load_video_tail(segment, frame_count):
            return video_map.get(segment.get("videoFile"), torch.zeros((1, 96, 160, 3), dtype=torch.float32))

        with (
            mock.patch.object(ltx_director, "_load_image_tensor", side_effect=fake_load_image),
            mock.patch.object(ltx_director, "_load_video_tail_tensor", side_effect=fake_load_video_tail),
            mock.patch.object(ltx_director, "_encode_relay", return_value=("patched", "conditioning")),
            mock.patch.object(ltx_director, "_build_combined_audio", return_value=None),
            mock.patch.object(ltx_director, "_load_source_video_outputs", return_value=(None, None, 0.0, 0)),
        ):
            result = ltx_director.LTXDirector.execute(
                model="model",
                clip="clip",
                global_prompt="",
                duration_frames=duration_frames,
                duration_seconds=1.0,
                timeline_data=json.dumps(timeline),
                local_prompts=local_prompts,
                segment_lengths=segment_lengths,
                guide_strength=guide_strength,
                aspect_ratio="16:9",
                orientation="landscape",
                quality_tier="1 - fast samples",
                use_input_image_size=use_input_image_size,
                resize_method=resize_method,
                optional_latent=optional_latent,
            )
        return result if return_full_result else result[4]

    def test_reference_images_keep_source_aspect_ratio(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_image_frames(
            tensor,
            target_w=320,
            target_h=160,
            derived_w=0,
            derived_h=0,
            use_input_image_size=False,
            divisible_by=32,
        )

        self.assertEqual(tuple(resized.shape), (1, 192, 96, 3))
        self.assertGreater(float(resized.mean()), 0.9)

    def test_reference_guides_are_padded_to_video_ratio(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_guide_frames(
            tensor,
            target_w=320,
            target_h=160,
            derived_w=0,
            derived_h=0,
            use_input_image_size=False,
            divisible_by=32,
        )

        self.assertEqual(tuple(resized.shape), (1, 160, 320, 3))
        self.assertTrue(torch.allclose(resized[:, :, :120, :], torch.zeros_like(resized[:, :, :120, :])))
        self.assertGreater(float(resized[:, :, 120:200, :].mean()), 0.9)
        self.assertTrue(torch.allclose(resized[:, :, 200:, :], torch.zeros_like(resized[:, :, 200:, :])))

    def test_padded_reference_inner_image_keeps_aspect_without_divisible_snap(self):
        tensor = torch.ones((1, 150, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_guide_frames(
            tensor,
            target_w=576,
            target_h=320,
            derived_w=0,
            derived_h=0,
            use_input_image_size=False,
            divisible_by=32,
        )

        expected_inner_w = 213
        expected_left = 181
        expected_right = expected_left + expected_inner_w

        self.assertEqual(tuple(resized.shape), (1, 320, 576, 3))
        self.assertTrue(torch.allclose(resized[:, :, :expected_left, :], torch.zeros_like(resized[:, :, :expected_left, :])))
        self.assertGreater(float(resized[:, :, expected_left:expected_right, :].mean()), 0.9)
        self.assertTrue(torch.allclose(resized[:, :, expected_right:, :], torch.zeros_like(resized[:, :, expected_right:, :])))
        self.assertLess(abs((expected_inner_w / 320) - (100 / 150)), 0.005)

    def test_timeline_resize_can_still_crop(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_image_frames(tensor, 320, 160, "crop", 32)

        self.assertEqual(tuple(resized.shape), (1, 160, 320, 3))
        self.assertGreater(float(resized.mean()), 0.9)

    def test_input_size_references_ignore_derived_video_dimensions(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_image_frames(
            tensor,
            target_w=320,
            target_h=160,
            derived_w=256,
            derived_h=256,
            use_input_image_size=True,
            divisible_by=32,
        )

        self.assertEqual(tuple(resized.shape), (1, 192, 96, 3))

    def test_input_size_reference_guides_pad_to_derived_dimensions(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_guide_frames(
            tensor,
            target_w=320,
            target_h=160,
            derived_w=256,
            derived_h=256,
            use_input_image_size=True,
            divisible_by=32,
        )

        self.assertEqual(tuple(resized.shape), (1, 256, 256, 3))
        self.assertTrue(torch.allclose(resized[:, :, :64, :], torch.zeros_like(resized[:, :, :64, :])))
        self.assertGreater(float(resized[:, :, 96:160, :].mean()), 0.9)

    def test_reference_only_input_size_reference_guides_fall_back_to_preset_dimensions(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_guide_frames(
            tensor,
            target_w=320,
            target_h=160,
            derived_w=0,
            derived_h=0,
            use_input_image_size=True,
            divisible_by=32,
        )

        self.assertEqual(tuple(resized.shape), (1, 160, 320, 3))

    def test_reference_segment_loader_uses_file_name_fallback(self):
        reference = torch.ones((1, 128, 256, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {
                    "id": "legacy-ref",
                    "label": "image1",
                    "kind": "character",
                    "fileName": "legacy-woman.png",
                    "strength": 0.8,
                },
            ],
            "segments": [
                {
                    "id": "uses-ref",
                    "type": "text",
                    "start": 0,
                    "length": 8,
                    "prompt": "@image1:character A young woman in a white dress enters from the left.",
                },
            ],
        }

        def fake_load_image(seg):
            if seg.get("imageFile") == "legacy-woman.png":
                return reference
            return torch.zeros((1, 32, 32, 3), dtype=torch.float32)

        with (
            mock.patch.object(ltx_director, "_load_image_tensor", side_effect=fake_load_image),
            mock.patch.object(ltx_director, "_encode_relay", return_value=("patched", "conditioning")),
            mock.patch.object(ltx_director, "_build_combined_audio", return_value=None),
            mock.patch.object(ltx_director, "_load_source_video_outputs", return_value=(None, None, 0.0, 0)),
        ):
            result = ltx_director.LTXDirector.execute(
                model="model",
                clip="clip",
                global_prompt="",
                duration_frames=8,
                duration_seconds=1.0,
                timeline_data=json.dumps(timeline),
                local_prompts="@image1:character A young woman in a white dress enters from the left.",
                segment_lengths="8",
                aspect_ratio="16:9",
                orientation="landscape",
                quality_tier="1 - fast samples",
            )

        guide_data = result[4]
        self.assertEqual(guide_data["reference_images"][0]["label"], "image1")
        self.assertGreater(float(guide_data["reference_images"][0]["image"].mean()), 0.5)

    def test_reference_only_input_size_keeps_source_aspect_ratio(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_reference_image_frames(
            tensor,
            target_w=320,
            target_h=160,
            derived_w=0,
            derived_h=0,
            use_input_image_size=True,
            divisible_by=32,
        )

        self.assertEqual(tuple(resized.shape), (1, 192, 96, 3))

    def test_character_references_populate_identity_metadata_and_normal_guides(self):
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {
                    "id": "ref-one",
                    "label": "image1",
                    "kind": "character",
                    "imageFile": "ref.png",
                    "strength": 0.7,
                    "description": "a young woman in a white dress",
                },
            ],
            "segments": [
                {
                    "id": "uses-ref",
                    "type": "text",
                    "start": 0,
                    "length": 8,
                    "prompt": "@image1:character A young woman in a white dress enters from the left.",
                },
            ],
        }

        encode_calls = []

        def fake_encode(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
            encode_calls.append(
                {
                    "global_prompt": global_prompt,
                    "local_prompts": local_prompts,
                    "segment_lengths": segment_lengths,
                }
            )
            return "patched", "conditioning"

        with (
            mock.patch.object(ltx_director, "_load_image_tensor", return_value=reference),
            mock.patch.object(ltx_director, "_encode_relay", side_effect=fake_encode),
            mock.patch.object(ltx_director, "_build_combined_audio", return_value=None),
            mock.patch.object(ltx_director, "_load_source_video_outputs", return_value=(None, None, 0.0, 0)),
        ):
            result = ltx_director.LTXDirector.execute(
                model="model",
                clip="clip",
                global_prompt="",
                duration_frames=8,
                duration_seconds=1.0,
                timeline_data=json.dumps(timeline),
                local_prompts="@image1:character A young woman in a white dress enters from the left.",
                segment_lengths="8",
                aspect_ratio="16:9",
                orientation="landscape",
                quality_tier="1 - fast samples",
            )

        guide_data = result[4]
        self.assertEqual(len(guide_data["reference_images"]), 1)
        self.assertEqual(guide_data["reference_images"][0]["label"], "image1")
        self.assertEqual(guide_data["reference_images"][0]["segment_id"], "uses-ref")
        self.assertEqual(guide_data["reference_images"][0]["strength"], 0.7)
        self.assertTrue(guide_data["reference_images"][0]["hidden_tail"])
        self.assertEqual(guide_data["reference_images"][0]["clean_latent_frames"], 2)
        self.assertEqual(guide_data["reference_images"][0]["clean_pixel_frames"], 9)
        self.assertEqual(guide_data["hidden_reference_count"], 1)
        self.assertEqual(guide_data["clean_latent_frames"], 2)
        self.assertEqual(guide_data["clean_pixel_frames"], 9)

        self.assertEqual(len(guide_data["images"]), 1)
        self.assertEqual(guide_data["insert_frames"], [16])
        self.assertEqual(guide_data["strengths"], [0.7])
        self.assertEqual(tuple(guide_data["images"][0].shape), (1, 320, 576, 3))
        self.assertTrue(torch.allclose(guide_data["images"][0][:, :, :192, :], torch.zeros_like(guide_data["images"][0][:, :, :192, :])))
        self.assertGreater(float(guide_data["images"][0][:, :, 224:352, :].mean()), 0.9)
        self.assertNotEqual(tuple(guide_data["images"][0].shape), tuple(guide_data["reference_images"][0]["image"].shape))
        self.assertEqual(len(encode_calls), 1)
        self.assertEqual(
            encode_calls[0]["local_prompts"],
            "a young woman in a white dress A young woman in a white dress enters from the left.",
        )

    def test_timeline_images_still_populate_normal_guides_with_references_present(self):
        timeline_image = torch.full((1, 160, 320, 3), 0.25, dtype=torch.float32)
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "ref.png"},
            ],
            "segments": [
                {
                    "id": "scene",
                    "type": "image",
                    "start": 0,
                    "length": 8,
                    "imageFile": "scene.png",
                    "prompt": "The seated couple waits.",
                },
                {
                    "id": "uses-ref",
                    "type": "text",
                    "start": 8,
                    "length": 8,
                    "prompt": "@image1:character A young woman in a white dress enters from the left.",
                },
            ],
        }

        def fake_load_image(segment):
            if segment.get("imageFile") == "scene.png":
                return timeline_image
            return reference

        with (
            mock.patch.object(ltx_director, "_load_image_tensor", side_effect=fake_load_image),
            mock.patch.object(ltx_director, "_encode_relay", return_value=("patched", "conditioning")),
            mock.patch.object(ltx_director, "_build_combined_audio", return_value=None),
            mock.patch.object(ltx_director, "_load_source_video_outputs", return_value=(None, None, 0.0, 0)),
        ):
            result = ltx_director.LTXDirector.execute(
                model="model",
                clip="clip",
                global_prompt="",
                duration_frames=16,
                duration_seconds=1.0,
                timeline_data=json.dumps(timeline),
                local_prompts="The seated couple waits. | @image1:character A young woman in a white dress enters from the left.",
                segment_lengths="8,8",
                guide_strength="0.4",
                aspect_ratio="16:9",
                orientation="landscape",
                quality_tier="1 - fast samples",
            )

        guide_data = result[4]
        self.assertEqual(len(guide_data["reference_images"]), 1)
        self.assertEqual(len(guide_data["images"]), 2)
        self.assertEqual(guide_data["insert_frames"], [0, 24])
        self.assertEqual(guide_data["strengths"], [0.4, 1.0])
        self.assertEqual(guide_data["hidden_reference_count"], 1)
        self.assertTrue(guide_data["reference_images"][0]["hidden_tail"])
        self.assertGreater(float(guide_data["images"][0].mean()), 0.2)
        self.assertEqual(tuple(guide_data["images"][1].shape), (1, 320, 576, 3))

    def test_unknown_global_reference_tag_raises_clear_error(self):
        timeline = {
            "referenceImages": [],
            "segments": [
                {"id": "scene", "type": "text", "start": 0, "length": 8, "prompt": "Scene"},
            ],
        }

        with self.assertRaises(ltx_director.LTXDirectorReferenceError):
            ltx_director.LTXDirector.execute(
                model="model",
                clip="clip",
                global_prompt="@image1:character",
                duration_frames=8,
                duration_seconds=1.0,
                timeline_data=json.dumps(timeline),
                local_prompts="Scene",
                segment_lengths="8",
                aspect_ratio="16:9",
                orientation="landscape",
                quality_tier="1 - fast samples",
            )

    def test_repeated_character_reference_uses_one_hidden_tail_slot(self):
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "ref.png"},
            ],
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "@image1:character enters"},
                {"id": "b", "type": "text", "start": 8, "length": 8, "prompt": "@image1:character turns"},
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"ref.png": reference},
            duration_frames=16,
            local_prompts="@image1:character enters | @image1:character turns",
            segment_lengths="8,8",
        )

        self.assertEqual(guide_data["hidden_reference_count"], 1)
        self.assertEqual(len(guide_data["reference_images"]), 1)
        self.assertEqual(guide_data["reference_images"][0]["segment_id"], "a,b")
        self.assertEqual(guide_data["insert_frames"], [24])

    def test_two_character_references_use_two_hidden_tail_slots(self):
        reference_one = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        reference_two = torch.full((1, 200, 100, 3), 0.5, dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "one.png"},
                {"id": "ref-two", "label": "image2", "kind": "character", "imageFile": "two.png"},
            ],
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "@image1:character enters"},
                {"id": "b", "type": "text", "start": 8, "length": 8, "prompt": "@image2:character turns"},
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"one.png": reference_one, "two.png": reference_two},
            duration_frames=16,
            local_prompts="@image1:character enters | @image2:character turns",
            segment_lengths="8,8",
        )

        self.assertEqual(guide_data["hidden_reference_count"], 2)
        self.assertEqual([ref["label"] for ref in guide_data["reference_images"]], ["image1", "image2"])
        self.assertEqual(guide_data["insert_frames"], [24, 32])

    def test_hidden_references_extend_generated_latent(self):
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "ref.png"},
            ],
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "@image1:character enters"},
            ],
        }

        result = self._execute_director_for_guide_data(
            timeline,
            {"ref.png": reference},
            duration_frames=8,
            local_prompts="@image1:character enters",
            segment_lengths="8",
            return_full_result=True,
        )

        latent = result[2]
        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 3, 10, 18))

    def test_no_character_references_keep_generated_latent_length(self):
        timeline = {
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "A quiet room."},
            ],
        }

        result = self._execute_director_for_guide_data(
            timeline,
            {},
            duration_frames=8,
            local_prompts="A quiet room.",
            segment_lengths="8",
            return_full_result=True,
        )

        guide_data = result[4]
        latent = result[2]
        self.assertEqual(guide_data["hidden_reference_count"], 0)
        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 2, 10, 18))

    def test_reference_tags_are_stripped_before_encoding(self):
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "ref.png"},
            ],
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "@image1:character enters"},
            ],
        }

        seen = {}

        def fake_encode(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
            seen["local_prompts"] = local_prompts
            return "patched", "conditioning"

        with (
            mock.patch.object(ltx_director, "_load_image_tensor", return_value=reference),
            mock.patch.object(ltx_director, "_encode_relay", side_effect=fake_encode),
            mock.patch.object(ltx_director, "_build_combined_audio", return_value=None),
            mock.patch.object(ltx_director, "_load_source_video_outputs", return_value=(None, None, 0.0, 0)),
        ):
            ltx_director.LTXDirector.execute(
                model="model",
                clip="clip",
                global_prompt="",
                duration_frames=8,
                duration_seconds=1.0,
                timeline_data=json.dumps(timeline),
                local_prompts="@image1:character enters",
                segment_lengths="8",
                aspect_ratio="16:9",
                orientation="landscape",
                quality_tier="1 - fast samples",
            )

        self.assertEqual(seen["local_prompts"], "enters")

    def test_unknown_character_reference_tag_raises_clear_error(self):
        timeline = {
            "referenceImages": [],
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "@image1:character enters"},
            ],
        }

        with (
            mock.patch.object(ltx_director, "_encode_relay", return_value=("patched", "conditioning")),
            mock.patch.object(ltx_director, "_build_combined_audio", return_value=None),
            mock.patch.object(ltx_director, "_load_source_video_outputs", return_value=(None, None, 0.0, 0)),
        ):
            with self.assertRaisesRegex(ValueError, "without matching enabled character reference images"):
                ltx_director.LTXDirector.execute(
                    model="model",
                    clip="clip",
                    global_prompt="",
                    duration_frames=8,
                    duration_seconds=1.0,
                    timeline_data=json.dumps(timeline),
                    local_prompts="@image1:character enters",
                    segment_lengths="8",
                    aspect_ratio="16:9",
                    orientation="landscape",
                    quality_tier="1 - fast samples",
                )

    def test_optional_latent_is_padded_for_hidden_references(self):
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        optional_latent = {
            "samples": torch.ones((1, 128, 2, 10, 18), dtype=torch.float32),
            "noise_mask": torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32),
        }
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "ref.png"},
            ],
            "segments": [
                {"id": "a", "type": "text", "start": 0, "length": 8, "prompt": "@image1:character enters"},
            ],
        }

        result = self._execute_director_for_guide_data(
            timeline,
            {"ref.png": reference},
            duration_frames=8,
            local_prompts="@image1:character enters",
            segment_lengths="8",
            optional_latent=optional_latent,
            return_full_result=True,
        )

        latent = result[2]
        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 3, 10, 18))
        self.assertTrue(torch.equal(latent["samples"][:, :, :2], optional_latent["samples"]))
        self.assertTrue(torch.equal(latent["samples"][:, :, 2:], torch.zeros_like(latent["samples"][:, :, 2:])))
        self.assertEqual(tuple(latent["noise_mask"].shape), (1, 1, 3, 1, 1))
        self.assertTrue(torch.equal(latent["noise_mask"][:, :, 2:], torch.ones_like(latent["noise_mask"][:, :, 2:])))

    def test_timeline_image_identity_fallback_uses_single_in_duration_image(self):
        timeline_image = torch.full((1, 160, 320, 3), 0.25, dtype=torch.float32)
        timeline = {
            "segments": [
                {
                    "id": "scene",
                    "type": "image",
                    "start": 0,
                    "length": 8,
                    "imageFile": "scene.png",
                    "prompt": "The seated couple waits.",
                },
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"scene.png": timeline_image},
            duration_frames=8,
            local_prompts="The seated couple waits.",
            segment_lengths="8",
            guide_strength="0.4",
        )

        self.assertEqual(len(guide_data["reference_images"]), 1)
        fallback = guide_data["reference_images"][0]
        self.assertEqual(fallback["id"], "timeline-images")
        self.assertEqual(fallback["label"], "image1")
        self.assertEqual(fallback["kind"], "timeline_image")
        self.assertEqual(fallback["segment_id"], "scene")
        self.assertEqual(fallback["insert_frame"], 0)
        self.assertEqual(tuple(fallback["image"].shape), tuple(guide_data["images"][0].shape))
        self.assertTrue(torch.equal(fallback["image"], guide_data["images"][0]))
        self.assertEqual(len(guide_data["images"]), 1)
        self.assertEqual(guide_data["insert_frames"], [0])
        self.assertEqual(guide_data["strengths"], [0.4])

    def test_timeline_image_identity_fallback_batches_multiple_in_duration_images(self):
        image_one = torch.full((1, 160, 320, 3), 0.25, dtype=torch.float32)
        image_two = torch.full((1, 200, 100, 3), 0.75, dtype=torch.float32)
        timeline = {
            "segments": [
                {"id": "scene-a", "type": "image", "start": 0, "length": 8, "imageFile": "a.png", "prompt": "A"},
                {"id": "scene-b", "type": "image", "start": 8, "length": 8, "imageFile": "b.png", "prompt": "B"},
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"a.png": image_one, "b.png": image_two},
            duration_frames=16,
            local_prompts="A | B",
            segment_lengths="8,8",
            guide_strength="0.4,0.5",
        )

        self.assertEqual(len(guide_data["reference_images"]), 1)
        fallback = guide_data["reference_images"][0]
        self.assertEqual(fallback["segment_id"], "scene-a,scene-b")
        self.assertEqual(tuple(fallback["image"].shape), (2, 320, 576, 3))
        self.assertEqual(len(guide_data["images"]), 2)
        self.assertEqual(guide_data["insert_frames"], [0, 8])
        self.assertEqual(guide_data["strengths"], [0.4, 0.5])
        self.assertNotEqual(tuple(guide_data["images"][0].shape), tuple(guide_data["images"][1].shape))

    def test_configured_character_references_suppress_timeline_identity_fallback(self):
        timeline_image = torch.full((1, 160, 320, 3), 0.25, dtype=torch.float32)
        reference = torch.ones((1, 200, 100, 3), dtype=torch.float32)
        timeline = {
            "referenceImages": [
                {"id": "ref-one", "label": "image1", "kind": "character", "imageFile": "ref.png"},
            ],
            "segments": [
                {"id": "scene", "type": "image", "start": 0, "length": 8, "imageFile": "scene.png", "prompt": "Scene"},
                {"id": "uses-ref", "type": "text", "start": 8, "length": 8, "prompt": "@image1:character enters"},
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"scene.png": timeline_image, "ref.png": reference},
            duration_frames=16,
            local_prompts="Scene | @image1:character enters",
            segment_lengths="8,8",
        )

        self.assertEqual(len(guide_data["reference_images"]), 1)
        self.assertEqual(guide_data["reference_images"][0]["id"], "ref-one")
        self.assertEqual(guide_data["reference_images"][0]["kind"], "character")

    def test_source_video_frames_are_excluded_from_timeline_identity_fallback(self):
        timeline_image = torch.full((1, 160, 320, 3), 0.25, dtype=torch.float32)
        source_video = torch.full((1, 96, 160, 3), 0.85, dtype=torch.float32)
        timeline = {
            "segments": [
                {"id": "video", "type": "source_video", "start": 0, "length": 8, "videoFile": "source.mp4", "prompt": ""},
                {"id": "scene", "type": "image", "start": 8, "length": 8, "imageFile": "scene.png", "prompt": "Scene"},
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"scene.png": timeline_image},
            video_map={"source.mp4": source_video},
            duration_frames=16,
            local_prompts="Scene",
            segment_lengths="16",
        )

        self.assertEqual(len(guide_data["images"]), 2)
        self.assertEqual(len(guide_data["reference_images"]), 1)
        fallback = guide_data["reference_images"][0]
        self.assertEqual(fallback["segment_id"], "scene")
        self.assertEqual(fallback["insert_frame"], 8)
        self.assertEqual(tuple(fallback["image"].shape), tuple(guide_data["images"][1].shape))

    def test_timeline_identity_fallback_excludes_images_outside_duration(self):
        in_duration = torch.full((1, 160, 320, 3), 0.25, dtype=torch.float32)
        outside = torch.full((1, 160, 320, 3), 0.75, dtype=torch.float32)
        timeline = {
            "segments": [
                {"id": "inside", "type": "image", "start": 0, "length": 8, "imageFile": "inside.png", "prompt": "Inside"},
                {"id": "outside", "type": "image", "start": 16, "length": 8, "imageFile": "outside.png", "prompt": "Outside"},
            ],
        }

        guide_data = self._execute_director_for_guide_data(
            timeline,
            {"inside.png": in_duration, "outside.png": outside},
            duration_frames=8,
            local_prompts="Inside",
            segment_lengths="8",
        )

        self.assertEqual(len(guide_data["images"]), 1)
        self.assertEqual(len(guide_data["reference_images"]), 1)
        self.assertEqual(guide_data["reference_images"][0]["segment_id"], "inside")
        self.assertEqual(tuple(guide_data["reference_images"][0]["image"].shape), tuple(guide_data["images"][0].shape))

    def test_crop_reference_tail_uses_director_metadata(self):
        latent = {
            "samples": torch.arange(1 * 128 * 5 * 2 * 2, dtype=torch.float32).reshape(1, 128, 5, 2, 2),
            "noise_mask": torch.ones((1, 1, 5, 1, 1), dtype=torch.float32),
        }
        guide_data = {"clean_latent_frames": 3, "clean_pixel_frames": 17}

        cropped, clean_pixel_frames = ltx_director.LTXDirectorCropReferenceTail.execute(latent, guide_data)

        self.assertEqual(clean_pixel_frames, 17)
        self.assertEqual(tuple(cropped["samples"].shape), (1, 128, 3, 2, 2))
        self.assertEqual(tuple(cropped["noise_mask"].shape), (1, 1, 3, 1, 1))
        self.assertTrue(torch.equal(cropped["samples"], latent["samples"][:, :, :3]))

    def test_crop_reference_tail_uses_hidden_count_plus_extra_when_metadata_is_too_long(self):
        latent = {
            "samples": torch.zeros((1, 128, 5, 2, 2), dtype=torch.float32),
            "noise_mask": torch.ones((1, 1, 5, 1, 1), dtype=torch.float32),
        }
        guide_data = {"clean_latent_frames": 5, "clean_pixel_frames": 33, "hidden_reference_count": 1}

        cropped, clean_pixel_frames = ltx_director.LTXDirectorCropReferenceTail.execute(latent, guide_data)

        self.assertEqual(clean_pixel_frames, 33)
        self.assertEqual(tuple(cropped["samples"].shape), (1, 128, 2, 2, 2))
        self.assertEqual(tuple(cropped["noise_mask"].shape), (1, 1, 2, 1, 1))

    def test_crop_reference_tail_trims_two_extra_latents_for_hidden_references(self):
        latent = {
            "samples": torch.zeros((1, 128, 5, 2, 2), dtype=torch.float32),
            "noise_mask": torch.ones((1, 1, 5, 1, 1), dtype=torch.float32),
        }
        guide_data = {"clean_latent_frames": 4, "clean_pixel_frames": 25, "hidden_reference_count": 1}

        cropped, clean_pixel_frames = ltx_director.LTXDirectorCropReferenceTail.execute(latent, guide_data)

        self.assertEqual(clean_pixel_frames, 25)
        self.assertEqual(tuple(cropped["samples"].shape), (1, 128, 2, 2, 2))
        self.assertEqual(tuple(cropped["noise_mask"].shape), (1, 1, 2, 1, 1))

    def test_crop_reference_tail_crops_nested_video_stream_only(self):
        video = torch.arange(1 * 128 * 5 * 2 * 2, dtype=torch.float32).reshape(1, 128, 5, 2, 2)
        audio = torch.ones((1, 32, 11, 4), dtype=torch.float32)
        video_mask = torch.ones((1, 1, 5, 1, 1), dtype=torch.float32)
        audio_mask = torch.ones_like(audio)
        latent = {
            "samples": FakeNestedTensor((video, audio)),
            "noise_mask": FakeNestedTensor((video_mask, audio_mask)),
        }
        guide_data = {"clean_latent_frames": 3, "clean_pixel_frames": 17}

        cropped, clean_pixel_frames = ltx_director.LTXDirectorCropReferenceTail.execute(latent, guide_data)
        cropped_video, cropped_audio = cropped["samples"].unbind()
        cropped_video_mask, cropped_audio_mask = cropped["noise_mask"].unbind()

        self.assertEqual(clean_pixel_frames, 17)
        self.assertIsInstance(cropped["samples"], FakeNestedTensor)
        self.assertEqual(tuple(cropped_video.shape), (1, 128, 3, 2, 2))
        self.assertTrue(torch.equal(cropped_video, video[:, :, :3]))
        self.assertIs(cropped_audio, audio)
        self.assertEqual(tuple(cropped_video_mask.shape), (1, 1, 3, 1, 1))
        self.assertTrue(torch.equal(cropped_video_mask, video_mask[:, :, :3]))
        self.assertIs(cropped_audio_mask, audio_mask)

    def test_crop_reference_tail_leaves_already_cropped_latent_shape_unchanged(self):
        latent = {
            "samples": torch.zeros((1, 128, 3, 2, 2), dtype=torch.float32),
            "noise_mask": torch.ones((1, 1, 3, 1, 1), dtype=torch.float32),
        }
        guide_data = {"clean_latent_frames": 3, "clean_pixel_frames": 17}

        cropped, clean_pixel_frames = ltx_director.LTXDirectorCropReferenceTail.execute(latent, guide_data)

        self.assertEqual(clean_pixel_frames, 17)
        self.assertEqual(tuple(cropped["samples"].shape), (1, 128, 3, 2, 2))
        self.assertEqual(tuple(cropped["noise_mask"].shape), (1, 1, 3, 1, 1))

    def test_crop_reference_tail_leaves_latent_unchanged_without_metadata(self):
        latent = {"samples": torch.zeros((1, 128, 5, 2, 2), dtype=torch.float32)}

        cropped, clean_pixel_frames = ltx_director.LTXDirectorCropReferenceTail.execute(latent, {})

        self.assertIs(cropped, latent)
        self.assertEqual(clean_pixel_frames, 0)


if __name__ == "__main__":
    unittest.main()
