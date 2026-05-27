import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_ltx_director():
    package_name = "wdc_audio_normalization_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(ROOT)
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
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
        Vae = _BaseType

        @staticmethod
        def Custom(_name):
            return _BaseType

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

    prompt_relay = types.ModuleType(f"{package_name}.prompt_relay")
    prompt_relay.get_raw_tokenizer = lambda _clip: None
    prompt_relay.map_token_indices = lambda *_args, **_kwargs: ("", [])
    prompt_relay.build_segments = lambda *_args, **_kwargs: []
    prompt_relay.create_mask_fn = lambda *_args, **_kwargs: None
    prompt_relay.distribute_segment_lengths = lambda *_args, **_kwargs: []
    sys.modules[prompt_relay.__name__] = prompt_relay

    patches = types.ModuleType(f"{package_name}.patches")
    patches.detect_model_type = lambda _model: ("ltx", (1, 1, 1), 1)
    patches.apply_patches = lambda *_args, **_kwargs: None
    sys.modules[patches.__name__] = patches

    image_config = types.ModuleType(f"{package_name}.timeline_image_config")
    image_config.resolve_image_path = lambda *_args: ""
    sys.modules[image_config.__name__] = image_config

    audio_config = types.ModuleType(f"{package_name}.timeline_audio_config")
    audio_config.resolve_audio_path = lambda *_args: ""
    sys.modules[audio_config.__name__] = audio_config

    privacy = types.ModuleType(f"{package_name}.ltx_director_privacy")
    privacy.resolve_ltx_director_inputs = lambda **kwargs: {
        key: kwargs[key]
        for key in ("global_prompt", "timeline_data", "local_prompts", "segment_lengths", "guide_strength")
    }
    sys.modules[privacy.__name__] = privacy

    module_name = f"{package_name}.ltx_director"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "ltx_director.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ltx_director = _load_ltx_director()


class AudioOutputNormalizationTests(unittest.TestCase):
    def test_disabled_normalization_leaves_waveform_unchanged(self):
        waveform = torch.tensor([[0.2, -0.4, 0.1], [0.3, -0.1, 0.0]], dtype=torch.float32)

        normalized = ltx_director._normalize_audio_waveform(waveform, enabled=False)

        self.assertTrue(torch.equal(normalized, waveform))

    def test_loud_waveform_is_reduced_below_peak_ceiling(self):
        waveform = torch.tensor([[1.4, -1.2, 0.9], [1.1, -1.3, 0.8]], dtype=torch.float32)

        normalized = ltx_director._normalize_audio_waveform(waveform)

        self.assertLessEqual(float(torch.max(torch.abs(normalized))), ltx_director.AUDIO_NORMALIZE_PEAK_CEILING + 1e-6)
        self.assertLess(float(torch.max(torch.abs(normalized))), float(torch.max(torch.abs(waveform))))

    def test_quiet_waveform_is_boosted_with_max_gain_cap(self):
        waveform = torch.full((2, 128), 0.01, dtype=torch.float32)

        normalized = ltx_director._normalize_audio_waveform(waveform)

        self.assertAlmostEqual(float(normalized[0, 0]), 0.02, places=6)

    def test_silent_waveform_is_unchanged(self):
        waveform = torch.zeros((2, 128), dtype=torch.float32)

        normalized = ltx_director._normalize_audio_waveform(waveform)

        self.assertTrue(torch.equal(normalized, waveform))


if __name__ == "__main__":
    unittest.main()
