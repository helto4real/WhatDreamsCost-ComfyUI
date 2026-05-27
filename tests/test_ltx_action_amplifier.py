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
        Clip = _BaseType
        Combo = _BaseType
        Conditioning = _BaseType
        Float = _BaseType
        Int = _BaseType
        Model = _BaseType
        String = _BaseType

        class Schema:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        @staticmethod
        def NodeOutput(*values):
            return values

    if "comfy_api.latest" in sys.modules:
        latest = sys.modules["comfy_api.latest"]
        if not hasattr(latest, "io"):
            latest.io = _Io
        for name in (
            "Boolean",
            "Clip",
            "Combo",
            "Conditioning",
            "Float",
            "Int",
            "Model",
            "String",
        ):
            if not hasattr(latest.io, name):
                setattr(latest.io, name, _BaseType)
        if not hasattr(latest.io, "Schema"):
            latest.io.Schema = _Io.Schema
        if not hasattr(latest.io, "NodeOutput"):
            latest.io.NodeOutput = _Io.NodeOutput
        return

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _Io
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest


_install_comfy_api_stub()
ltx_action_amplifier = importlib.import_module("ltx_action_amplifier")
vendor_action_amplifier = importlib.import_module("vendor.tenstrip_10s.latent_action_amplifier")


class _FakeRemoveHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeModel:
    def __init__(self, backbone, cloned=False):
        self.model = types.SimpleNamespace(diffusion_model=backbone)
        self.cloned = cloned

    def clone(self):
        return _FakeModel(self.model.diffusion_model, cloned=True)


class LTXActionAmplifierTests(unittest.TestCase):
    def test_schema_exposes_action_amplifier_contract(self):
        schema = ltx_action_amplifier.LTXActionAmplifier.define_schema()

        self.assertEqual(schema.node_id, "LTXActionAmplifier")
        self.assertEqual(schema.display_name, "LTX Action Amplifier")
        self.assertEqual(schema.category, "LTXVCustom/Conditioning")
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].kwargs["display_name"], "model")

    def test_execute_forwards_parameters_to_vendored_helper(self):
        seen = {}

        class FakeActionAmplifier:
            def apply(self, *args, **kwargs):
                seen["args"] = args
                seen["kwargs"] = kwargs
                return ("patched-model",)

        with mock.patch.object(ltx_action_amplifier, "_TenSActionAmplifier", FakeActionAmplifier):
            result = ltx_action_amplifier.LTXActionAmplifier.execute(
                "model",
                "clip",
                "positive",
                amplification_strength=0.45,
                action_vocabulary_text="walking, reaching",
                scale_ceiling=0.25,
                auto_threshold="p98",
                similarity_threshold=0.6,
                similarity_sharpness=20.0,
                amplification_floor=0.4,
                top_k=5,
                bypass=True,
                debug=True,
            )

        self.assertEqual(result, ("patched-model",))
        self.assertEqual(seen["args"], ("model", "clip", "positive", 0.45))
        self.assertEqual(
            seen["kwargs"],
            {
                "action_vocabulary_text": "walking, reaching",
                "scale_ceiling": 0.25,
                "auto_threshold": "p98",
                "similarity_threshold": 0.6,
                "similarity_sharpness": 20.0,
                "amplification_floor": 0.4,
                "top_k": 5,
                "bypass": True,
                "debug": True,
            },
        )

    def test_vendored_bypass_clones_model_and_cleans_prior_patch(self):
        original_forward = object()
        attn2 = types.SimpleNamespace(forward=object())
        setattr(attn2, vendor_action_amplifier.ORIGINAL_FORWARD_ATTR, original_forward)
        setattr(attn2, vendor_action_amplifier.HOOK_ATTR_ATTN2, True)
        block = types.SimpleNamespace(attn2=attn2)
        backbone = types.SimpleNamespace(transformer_blocks=[block])
        handle = _FakeRemoveHandle()
        setattr(backbone, "_10s_actamp_backbone_handle", handle)
        setattr(backbone, vendor_action_amplifier.HOOK_ATTR_BACKBONE, True)
        model = _FakeModel(backbone)

        result = vendor_action_amplifier.LTXActionAmplifier().apply(
            model,
            clip=None,
            positive=None,
            amplification_strength=0.0,
            bypass=True,
        )[0]

        self.assertIsNot(result, model)
        self.assertTrue(result.cloned)
        self.assertIs(attn2.forward, original_forward)
        self.assertFalse(hasattr(attn2, vendor_action_amplifier.ORIGINAL_FORWARD_ATTR))
        self.assertFalse(hasattr(attn2, vendor_action_amplifier.HOOK_ATTR_ATTN2))
        self.assertTrue(handle.removed)
        self.assertFalse(hasattr(backbone, "_10s_actamp_backbone_handle"))
        self.assertFalse(hasattr(backbone, vendor_action_amplifier.HOOK_ATTR_BACKBONE))


if __name__ == "__main__":
    unittest.main()
