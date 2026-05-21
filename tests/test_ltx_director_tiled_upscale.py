import importlib
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_comfy_stubs():
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
        Boolean = _BaseType
        Combo = _BaseType
        Conditioning = _BaseType
        Float = _BaseType
        Image = _BaseType
        Int = _BaseType
        Latent = _BaseType
        Model = _BaseType
        Sigmas = _BaseType
        String = _BaseType
        Vae = _BaseType

        @staticmethod
        def Custom(_name):
            return type("CustomType", (_BaseType,), {})

        class Schema:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def NodeOutput(*values):
            return values

    class _FakeLTXVAddGuide:
        encode_calls = []

        @classmethod
        def encode(cls, _vae, width, height, img_tensor, _scale_factors):
            cls.encode_calls.append((width, height, tuple(img_tensor.shape)))
            return img_tensor, torch.full((1, 128, 1, height, width), 0.5)

        @staticmethod
        def get_latent_index(_positive, _latent_length, _image_len, frame, scale_factors):
            return frame, frame // scale_factors[0]

        @staticmethod
        def append_keyframe(positive, negative, frame_idx, latent_image, noise_mask, encoded, strength, _scale_factors):
            latent_image[:, :, frame_idx:frame_idx + encoded.shape[2]] = encoded
            noise_mask[:, :, frame_idx:frame_idx + encoded.shape[2]] = 1.0 - strength
            return positive, negative, latent_image, noise_mask

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _Io
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    comfy_extras = types.ModuleType("comfy_extras")
    nodes_lt = types.ModuleType("comfy_extras.nodes_lt")
    nodes_lt.LTXVAddGuide = _FakeLTXVAddGuide
    sys.modules["comfy_extras"] = comfy_extras
    sys.modules["comfy_extras.nodes_lt"] = nodes_lt

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.module_size = lambda _model: 0
    model_management.free_memory = lambda *_args, **_kwargs: None
    comfy.model_management = model_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management

    return _FakeLTXVAddGuide


FakeLTXVAddGuide = _install_comfy_stubs()
ltx_tiled = importlib.import_module("ltx_director_tiled_upscale")


class _Stats:
    @staticmethod
    def un_normalize(latents):
        return latents

    @staticmethod
    def normalize(latents):
        return latents


class _FakeVAE:
    downscale_index_formula = (8, 32, 32)
    first_stage_model = types.SimpleNamespace(per_channel_statistics=_Stats())


class _DoubleUpscaler:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def __call__(self, latents):
        return latents.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)


class LTXDirectorTiledUpscaleTests(unittest.TestCase):
    def setUp(self):
        FakeLTXVAddGuide.encode_calls = []

    def test_settings_node_outputs_defaults(self):
        result = ltx_tiled.LTXDirectorTiledUpscaleSettings.execute()[0]

        self.assertEqual(
            result,
            {
                "tile_size": 24,
                "overlap": 8,
                "max_size_for_no_tile": 32,
                "rotate_for_landscape": False,
                "debug": False,
            },
        )

    def test_settings_node_outputs_custom_values(self):
        result = ltx_tiled.LTXDirectorTiledUpscaleSettings.execute(
            tile_size=16,
            overlap=4,
            max_size_for_no_tile=12,
            rotate_for_landscape=True,
            debug=True,
        )[0]

        self.assertEqual(result["tile_size"], 16)
        self.assertEqual(result["overlap"], 4)
        self.assertEqual(result["max_size_for_no_tile"], 12)
        self.assertTrue(result["rotate_for_landscape"])
        self.assertTrue(result["debug"])

    def test_tiled_upscale_guide_upscales_before_reapplying_guides(self):
        latent = {"samples": torch.zeros((1, 128, 2, 4, 5), dtype=torch.float32)}
        guide_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        positive, negative, out_latent = ltx_tiled.LTXDirectorTiledUpscaleGuide.execute(
            "positive",
            "negative",
            _FakeVAE(),
            latent,
            {"images": [guide_image], "insert_frames": [0], "strengths": [0.25]},
            _DoubleUpscaler(),
            {"tile_size": 3, "overlap": 1, "max_size_for_no_tile": 1},
        )

        self.assertEqual(positive, "positive")
        self.assertEqual(negative, "negative")
        self.assertEqual(tuple(out_latent["samples"].shape), (1, 128, 2, 8, 10))
        self.assertEqual(FakeLTXVAddGuide.encode_calls, [(10, 8, (1, 64, 64, 3))])
        self.assertTrue(torch.allclose(out_latent["samples"][:, :, 0], torch.full((1, 128, 8, 10), 0.5)))
        self.assertTrue(torch.allclose(out_latent["noise_mask"][:, :, 0], torch.full((1, 1, 1, 1), 0.75)))


if __name__ == "__main__":
    unittest.main()
