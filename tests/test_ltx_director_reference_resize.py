import importlib.util
import sys
import types
import unittest
from pathlib import Path

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


class LTXDirectorReferenceResizeTests(unittest.TestCase):
    def test_reference_images_are_padded_to_video_ratio(self):
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

        self.assertEqual(tuple(resized.shape), (1, 160, 320, 3))
        self.assertTrue(torch.allclose(resized[:, :, :96, :], torch.zeros_like(resized[:, :, :96, :])))
        self.assertGreater(float(resized[:, :, 128:192, :].mean()), 0.9)

    def test_timeline_resize_can_still_crop(self):
        tensor = torch.ones((1, 200, 100, 3), dtype=torch.float32)

        resized = ltx_director._resize_image_frames(tensor, 320, 160, "crop", 32)

        self.assertEqual(tuple(resized.shape), (1, 160, 320, 3))
        self.assertGreater(float(resized.mean()), 0.9)

    def test_input_size_references_pad_to_derived_dimensions(self):
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

        self.assertEqual(tuple(resized.shape), (1, 256, 256, 3))

    def test_reference_only_input_size_falls_back_to_preset_dimensions(self):
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

        self.assertEqual(tuple(resized.shape), (1, 160, 320, 3))


if __name__ == "__main__":
    unittest.main()
