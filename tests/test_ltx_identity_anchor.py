import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_comfy_api_stub():
    if "comfy_api.latest" in sys.modules:
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
        Model = _BaseType
        Image = _BaseType
        Latent = _BaseType
        Float = _BaseType
        Int = _BaseType
        Boolean = _BaseType
        String = _BaseType
        Combo = _BaseType
        Sigmas = _BaseType
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


_install_comfy_api_stub()
ltx_identity_anchor = importlib.import_module("ltx_identity_anchor")


class LTXIdentityAnchorTests(unittest.TestCase):
    def test_missing_identity_anchor_returns_original_model(self):
        model = object()

        result = ltx_identity_anchor.apply_identity_anchor(model)

        self.assertIs(result, model)

    def test_combined_anchor_orders_latent_aware_before_face_and_scales_strengths(self):
        calls = []

        class FakeAware:
            def patch(self, model, **kwargs):
                calls.append(("aware", kwargs["strength"]))
                return (model + ["aware"],)

        class FakeFace:
            def patch(self, model, *args, **kwargs):
                calls.append(("face", kwargs["strength"]))
                return (model + ["face"],)

        def fake_loader(filename, class_name):
            if filename == "latent_anchor_aware.py":
                return FakeAware
            if filename == "face_anchor.py":
                return FakeFace
            raise AssertionError((filename, class_name))

        identity_anchor = {
            "kind": "combined",
            "scale_strengths": True,
            "strength_scale": 0.5,
            "anchors": [
                {"kind": "face", "strength": 0.2},
                {"kind": "latent_aware", "strength": 0.1, "energy_source": "none"},
            ],
        }

        with mock.patch.object(ltx_identity_anchor, "_load_10s_class", side_effect=fake_loader):
            result = ltx_identity_anchor.apply_identity_anchor([], identity_anchor)

        self.assertEqual(result, ["aware", "face"])
        self.assertEqual(calls, [("aware", 0.05), ("face", 0.1)])

    def test_latent_aware_can_use_first_guide_image_as_reference(self):
        guide_image = object()
        seen = {}

        class FakeAware:
            def patch(self, model, **kwargs):
                seen.update(kwargs)
                return (model,)

        with mock.patch.object(ltx_identity_anchor, "_load_10s_class", return_value=FakeAware):
            ltx_identity_anchor.apply_identity_anchor(
                "model",
                {
                    "kind": "latent_aware",
                    "energy_source": "first_guide_image",
                    "strength": 0.1,
                },
                vae="vae",
                guide_data={"images": [guide_image]},
            )

        self.assertIs(seen["reference_image"], guide_image)
        self.assertIsNone(seen["energy_latent"])
        self.assertEqual(seen["vae"], "vae")


if __name__ == "__main__":
    unittest.main()
