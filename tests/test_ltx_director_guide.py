import unittest
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_comfy_stubs():
    if "comfy_api.latest" in sys.modules and "comfy_extras.nodes_lt" in sys.modules:
        return

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
        Conditioning = _BaseType
        Float = _BaseType
        Combo = _BaseType
        Latent = _BaseType
        Model = _BaseType
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

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _Io
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    nodes_lt = types.ModuleType("comfy_extras.nodes_lt")
    nodes_lt.ICLoRAParameters = _Io.Custom("IC_LORA_PARAMETERS")
    nodes_lt.LTXVAddGuide = object
    nodes_lt._append_guide_attention_entry = lambda positive, negative, *args, **kwargs: (positive, negative)
    nodes_lt.get_keyframe_idxs = lambda positive, latent_shape: ([], 0)
    sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
    sys.modules["comfy_extras.nodes_lt"] = nodes_lt

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy.utils = comfy_utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils

    ltx_director = types.ModuleType("ltx_director")
    ltx_director.GuideData = _Io.Custom("GUIDE_DATA")
    sys.modules["ltx_director"] = ltx_director


_install_comfy_stubs()
from ltx_director_guide import extract_director_iclora_parameters


class FakeModel:
    def __init__(self, metadata):
        self.metadata = metadata

    def get_attachment(self, name):
        if name == "lora_metadata":
            return self.metadata
        return None


class LTXDirectorGuideICLoRATests(unittest.TestCase):
    def test_extracts_reference_downscale_factor(self):
        params = extract_director_iclora_parameters(FakeModel({"reference_downscale_factor": "2"}))

        self.assertEqual(params["reference_downscale_factor"], 2)
        self.assertTrue(params["metadata_present"])

    def test_errors_when_metadata_is_missing(self):
        with self.assertRaisesRegex(ValueError, "does not expose IC-LoRA metadata"):
            extract_director_iclora_parameters(FakeModel(None))

    def test_errors_when_reference_downscale_factor_is_missing(self):
        with self.assertRaisesRegex(ValueError, "does not include reference_downscale_factor"):
            extract_director_iclora_parameters(FakeModel({"other": "1"}))

    def test_errors_when_reference_downscale_factor_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "invalid reference_downscale_factor"):
            extract_director_iclora_parameters(FakeModel({"reference_downscale_factor": "nope"}))


if __name__ == "__main__":
    unittest.main()
