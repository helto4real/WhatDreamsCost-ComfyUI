from comfy_extras.nodes_lt import LTXVAddGuide
from comfy_api.latest import io
import torch

try:
    from .ltx_director import GuideData
    from .vendor.tenstrip_10s.latent_upsampler_tiled import LTXVLatentUpsamplerTiled
except Exception:  # noqa: BLE001 - allow direct module imports in tests.
    try:
        from ltx_director import GuideData
    except Exception:  # noqa: BLE001
        GuideData = io.Custom("GUIDE_DATA")
    from vendor.tenstrip_10s.latent_upsampler_tiled import LTXVLatentUpsamplerTiled


TiledUpscaleSettings = io.Custom("LTX_TILED_UPSCALE_SETTINGS")
LatentUpscaleModel = io.Custom("LATENT_UPSCALE_MODEL")


DEFAULT_TILED_UPSCALE_SETTINGS = {
    "tile_size": 24,
    "overlap": 8,
    "max_size_for_no_tile": 32,
    "rotate_for_landscape": False,
    "debug": False,
}


def _normalize_settings(settings):
    normalized = DEFAULT_TILED_UPSCALE_SETTINGS.copy()
    if isinstance(settings, dict):
        normalized.update({key: settings[key] for key in normalized if key in settings})
    normalized["tile_size"] = int(normalized["tile_size"])
    normalized["overlap"] = int(normalized["overlap"])
    normalized["max_size_for_no_tile"] = int(normalized["max_size_for_no_tile"])
    normalized["rotate_for_landscape"] = bool(normalized["rotate_for_landscape"])
    normalized["debug"] = bool(normalized["debug"])
    return normalized


def _apply_director_guides(node_cls, positive, negative, vae, latent, guide_data):
    scale_factors = vae.downscale_index_formula
    latent_image = latent["samples"].clone()

    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"].clone()
    else:
        batch, _, latent_frames, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch, 1, latent_frames, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )

    _, _, latent_length, latent_height, latent_width = latent_image.shape
    images = guide_data.get("images", []) if isinstance(guide_data, dict) else []
    insert_frames = guide_data.get("insert_frames", []) if isinstance(guide_data, dict) else []
    strengths = guide_data.get("strengths", []) if isinstance(guide_data, dict) else []

    for idx, img_tensor in enumerate(images):
        frame = insert_frames[idx] if idx < len(insert_frames) else 0
        strength = strengths[idx] if idx < len(strengths) else 1.0

        image_1, encoded = node_cls.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
        frame_idx, latent_idx = node_cls.get_latent_index(positive, latent_length, len(image_1), frame, scale_factors)
        if latent_idx + encoded.shape[2] > latent_length:
            raise AssertionError(f"Guide image {idx + 1}: conditioning frames exceed the length of the latent sequence.")

        positive, negative, latent_image, noise_mask = node_cls.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent_image,
            noise_mask,
            encoded,
            strength,
            scale_factors,
        )

    return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


class LTXDirectorTiledUpscaleSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorTiledUpscaleSettings",
            display_name="LTX Director Tiled Upscale Settings",
            category="LTXVCustom/Upscale",
            description="Settings for LTX Director tiled latent upscaling.",
            inputs=[
                io.Int.Input("tile_size", default=24, min=8, max=128, step=1),
                io.Int.Input("overlap", default=8, min=2, max=32, step=1),
                io.Int.Input("max_size_for_no_tile", default=32, min=8, max=256, step=1),
                io.Boolean.Input("rotate_for_landscape", default=False),
                io.Boolean.Input("debug", default=False),
            ],
            outputs=[
                TiledUpscaleSettings.Output(display_name="upscale_settings"),
            ],
        )

    @classmethod
    def execute(
        cls,
        tile_size=24,
        overlap=8,
        max_size_for_no_tile=32,
        rotate_for_landscape=False,
        debug=False,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            _normalize_settings(
                {
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "max_size_for_no_tile": max_size_for_no_tile,
                    "rotate_for_landscape": rotate_for_landscape,
                    "debug": debug,
                }
            )
        )


class LTXDirectorTiledUpscaleGuide(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorTiledUpscaleGuide",
            display_name="LTX Director Tiled Upscale Guide",
            category="LTXVCustom",
            description=(
                "Runs the LTX tiled latent upscaler, then reapplies LTX Director guide images "
                "at their timeline frame positions for phase-two refinement."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to add guide keyframe info to."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to add guide keyframe info to."),
                io.Vae.Input("vae", tooltip="Video VAE used by the tiled upscaler and guide image encoder."),
                io.Latent.Input("latent", tooltip="Phase-one video latent, usually after LTXVCropGuides."),
                GuideData.Input("guide_data", tooltip="Guide data produced by LTX Director."),
                LatentUpscaleModel.Input("upscale_model", tooltip="LTX latent upscale model."),
                TiledUpscaleSettings.Input("upscale_settings", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Upscaled video latent with Director guides reapplied."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data, upscale_model, upscale_settings=None) -> io.NodeOutput:
        settings = _normalize_settings(upscale_settings)
        upscaled_latent = LTXVLatentUpsamplerTiled().upsample_latent_tiled(
            latent,
            upscale_model,
            vae,
            tile_size=settings["tile_size"],
            overlap=settings["overlap"],
            max_size_for_no_tile=settings["max_size_for_no_tile"],
            rotate_for_landscape=settings["rotate_for_landscape"],
            debug=settings["debug"],
        )[0]

        positive, negative, guided_latent = _apply_director_guides(
            cls,
            positive,
            negative,
            vae,
            upscaled_latent,
            guide_data,
        )
        return io.NodeOutput(positive, negative, guided_latent)


NODE_CLASS_MAPPINGS = {
    "LTXDirectorTiledUpscaleSettings": LTXDirectorTiledUpscaleSettings,
    "LTXDirectorTiledUpscaleGuide": LTXDirectorTiledUpscaleGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirectorTiledUpscaleSettings": "LTX Director Tiled Upscale Settings",
    "LTXDirectorTiledUpscaleGuide": "LTX Director Tiled Upscale Guide",
}
