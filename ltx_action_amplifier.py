from comfy_api.latest import io

try:
    from .vendor.tenstrip_10s.latent_action_amplifier import (
        DEFAULT_ACTION_VOCABULARY,
        LTXActionAmplifier as _TenSActionAmplifier,
    )
except Exception:  # noqa: BLE001 - allow direct module imports in tests.
    from vendor.tenstrip_10s.latent_action_amplifier import (
        DEFAULT_ACTION_VOCABULARY,
        LTXActionAmplifier as _TenSActionAmplifier,
    )


class LTXActionAmplifier(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXActionAmplifier",
            display_name="LTX Action Amplifier",
            category="LTXVCustom/Conditioning",
            description=(
                "Selectively amplifies action and motion tokens in positive "
                "conditioning using the bundled 10S Action Amplifier."
            ),
            inputs=[
                io.Model.Input("model", tooltip="LTX model to patch before sampling."),
                io.Clip.Input("clip", tooltip="Text encoder used to encode the action vocabulary."),
                io.Conditioning.Input("positive", tooltip="Positive conditioning to analyze and match at runtime."),
                io.Float.Input(
                    "amplification_strength",
                    default=0.30,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="How strongly to boost matched action tokens. 0.30 is a gentle starting point.",
                ),
                io.String.Input(
                    "action_vocabulary_text",
                    default=DEFAULT_ACTION_VOCABULARY,
                    multiline=True,
                    optional=True,
                    tooltip="Comma-separated action or motion vocabulary used for token matching.",
                ),
                io.Float.Input(
                    "scale_ceiling",
                    default=0.30,
                    min=0.05,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="Maximum K/V scale delta. Default 0.30 caps matched tokens at +30%.",
                ),
                io.Combo.Input(
                    "auto_threshold",
                    options=["disabled", "p90", "p95", "p98", "p99"],
                    default="p95",
                    optional=True,
                    tooltip="Percentile threshold for selecting action-like tokens.",
                ),
                io.Float.Input(
                    "similarity_threshold",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Literal threshold used when auto_threshold is disabled.",
                ),
                io.Float.Input(
                    "similarity_sharpness",
                    default=16.0,
                    min=1.0,
                    max=64.0,
                    step=0.5,
                    optional=True,
                    tooltip="Sigmoid steepness for token selection.",
                ),
                io.Float.Input(
                    "amplification_floor",
                    default=0.30,
                    min=0.0,
                    max=0.9,
                    step=0.05,
                    optional=True,
                    tooltip="Hard floor for weak token matches; below this value no boost is applied.",
                ),
                io.Int.Input(
                    "top_k",
                    default=3,
                    min=1,
                    max=16,
                    step=1,
                    optional=True,
                    tooltip="Number of top vocabulary similarities averaged per positive token.",
                ),
                io.Boolean.Input(
                    "bypass",
                    default=False,
                    optional=True,
                    tooltip="Pass through unchanged and clear prior Action Amplifier patches.",
                ),
                io.Boolean.Input("debug", default=False, optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        positive,
        amplification_strength=0.30,
        action_vocabulary_text=DEFAULT_ACTION_VOCABULARY,
        scale_ceiling=0.30,
        auto_threshold="p95",
        similarity_threshold=0.55,
        similarity_sharpness=16.0,
        amplification_floor=0.30,
        top_k=3,
        bypass=False,
        debug=False,
    ) -> io.NodeOutput:
        patched_model = _TenSActionAmplifier().apply(
            model,
            clip,
            positive,
            amplification_strength,
            action_vocabulary_text=action_vocabulary_text,
            scale_ceiling=scale_ceiling,
            auto_threshold=auto_threshold,
            similarity_threshold=similarity_threshold,
            similarity_sharpness=similarity_sharpness,
            amplification_floor=amplification_floor,
            top_k=top_k,
            bypass=bypass,
            debug=debug,
        )[0]
        return io.NodeOutput(patched_model)


NODE_CLASS_MAPPINGS = {
    "LTXActionAmplifier": LTXActionAmplifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXActionAmplifier": "LTX Action Amplifier",
}
