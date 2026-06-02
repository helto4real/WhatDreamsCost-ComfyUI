from comfy_extras.nodes_lt import ICLoRAParameters, LTXVAddGuide, _append_guide_attention_entry, get_keyframe_idxs
import torch
import comfy.utils
from comfy_api.latest import io
try:
    from .ltx_director import GuideData
except ImportError:
    from ltx_director import GuideData


def extract_director_iclora_parameters(iclora_model):
    try:
        metadata = iclora_model.get_attachment("lora_metadata")
    except AttributeError as exc:
        raise ValueError(
            "LTX Director IC-LoRA parameters must be extracted from the MODEL output of a ComfyUI Load LoRA node."
        ) from exc

    if not metadata:
        raise ValueError(
            "The connected LoRA does not expose IC-LoRA metadata. Use an IC-LoRA safetensors file that includes "
            "reference_downscale_factor metadata, and connect the MODEL output directly from the LoRA loader."
        )

    if "reference_downscale_factor" not in metadata:
        raise ValueError(
            "The connected LoRA metadata does not include reference_downscale_factor, so LTX Director cannot "
            "safely apply IC-LoRA guide handling for it."
        )

    try:
        factor = max(1, round(float(metadata.get("reference_downscale_factor"))))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The connected LoRA has invalid reference_downscale_factor metadata. Expected a numeric value."
        ) from exc

    return {"reference_downscale_factor": factor, "metadata_present": True}


class LTXDirectorGetICLoRAParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorGetICLoRAParameters",
            display_name="LTX Director Get IC-LoRA Parameters",
            category="WhatDreamsCost",
            description=(
                "Extracts required IC-LoRA metadata from a LoRA-loaded model for LTX Director Guide. "
                "Raises an error if the LoRA does not include reference_downscale_factor metadata."
            ),
            inputs=[
                io.Model.Input(
                    "iclora_model",
                    tooltip="Direct MODEL output from the Load LoRA node that loaded the IC-LoRA.",
                ),
            ],
            outputs=[
                ICLoRAParameters.Output(
                    "iclora_parameters",
                    tooltip="Strict IC-LoRA parameters for LTX Director Guide.",
                ),
            ],
        )

    @classmethod
    def execute(cls, iclora_model) -> io.NodeOutput:
        return io.NodeOutput(extract_director_iclora_parameters(iclora_model))


class LTXDirectorGuide(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorGuide",
            display_name="LTX Director Guide",
            category="WhatDreamsCost",
            description=(
                "Applies guide images from a Prompt Relay Timeline node at the frame positions "
                "and strengths defined on the timeline. Connect guide_data from the timeline node."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to add guide keyframe info to."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to add guide keyframe info to."),
                io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images."),
                io.Latent.Input("latent", tooltip="Video latent — guides are inserted into this latent."),
                GuideData.Input("guide_data", tooltip="Guide data produced by Prompt Relay Encode (Timeline)."),
                ICLoRAParameters.Input(
                    "iclora_parameters",
                    optional=True,
                    tooltip=(
                        "Optional IC-LoRA parameters from LTX Director Get IC-LoRA Parameters. "
                        "Used for adjusting guide processing as required by certain IC-LoRAs "
                        "(for example reference_downscale_factor)."
                    ),
                ),
                io.Float.Input("scale_by", default=1.0, min=0.01, max=8.0, step=0.01, tooltip="Scale the latent by this factor."),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], default="bicubic", tooltip="Method used to upscale/downscale the latent."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Video latent with guide frames applied."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data, iclora_parameters=None, scale_by=1.0, upscale_method="bicubic") -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula

        # Clone latents to avoid mutating upstream nodes
        latent_image = latent["samples"].clone()
        latent_downscale_factor = cls.get_reference_downscale_factor(iclora_parameters)

        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, latent_height, latent_width = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        # Apply scale factor if not 1.0
        if scale_by != 1.0:
            B, C, F, H, W = latent_image.shape
            width = round(W * scale_by)
            height = round(H * scale_by)
            
            # Reshape to 4D for common_upscale
            latent_4d = latent_image.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            latent_resized_4d = comfy.utils.common_upscale(latent_4d, width, height, upscale_method, "disabled")
            latent_image = latent_resized_4d.reshape(B, F, C, height, width).permute(0, 2, 1, 3, 4)

            # Also resize noise mask if it's not a broadcasted mask
            if noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1:
                mask_4d = noise_mask.permute(0, 2, 1, 3, 4).reshape(B * F, 1, H, W)
                mask_resized_4d = comfy.utils.common_upscale(mask_4d, width, height, upscale_method, "disabled")
                noise_mask = mask_resized_4d.reshape(B, F, 1, height, width).permute(0, 2, 1, 3, 4)

        _, _, _, latent_height, latent_width = latent_image.shape
        if latent_downscale_factor > 1:
            if latent_width % latent_downscale_factor != 0 or latent_height % latent_downscale_factor != 0:
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                    f"reference_downscale_factor {latent_downscale_factor} from the IC-LoRA parameters."
                )

        images = guide_data.get("images", [])
        insert_frames = guide_data.get("insert_frames", [])
        strengths = guide_data.get("strengths", [])

        for idx, img_tensor in enumerate(images):
            f_idx = insert_frames[idx] if idx < len(insert_frames) else 0
            strength = strengths[idx] if idx < len(strengths) else 1.0
            _, _, latent_length, latent_height, latent_width = latent_image.shape

            # Match ComfyUI LTXVAddGuide handling for mid-video multi-frame guides:
            # use a throwaway first frame so VAE temporal asymmetry lands outside
            # the retained guide slot.
            time_scale_factor = scale_factors[0]
            num_frames_to_keep = ((img_tensor.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
            resolved_frame_idx = f_idx
            if f_idx < 0:
                _, num_keyframes = get_keyframe_idxs(positive, latent_image.shape)
                resolved_frame_idx = max((latent_length - num_keyframes - 1) * time_scale_factor + 1 + f_idx, 0)
            causal_fix = resolved_frame_idx == 0 or num_frames_to_keep == 1

            image = img_tensor
            if not causal_fix:
                image = torch.cat([image[:1], image], dim=0)

            image_1, t = cls.encode(vae, latent_width, latent_height, image, scale_factors, latent_downscale_factor)

            if not causal_fix:
                t = t[:, :, 1:, :, :]
                image_1 = image_1[1:]

            guide_latent_shape = list(t.shape[2:])
            guide_mask = None
            if latent_downscale_factor > 1:
                t, guide_mask = cls.dilate_latent(t, latent_downscale_factor)

            frame_idx, latent_idx = cls.get_latent_index(
                positive, latent_length, len(image_1), f_idx, scale_factors, latent_shape=latent_image.shape
            )

            assert latent_idx + t.shape[2] <= latent_length, (
                f"Guide image {idx + 1}: conditioning frames exceed the length of the latent sequence."
            )

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
                guide_mask=guide_mask,
                latent_downscale_factor=latent_downscale_factor,
                causal_fix=causal_fix,
            )

            pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
            positive, negative = _append_guide_attention_entry(
                positive,
                negative,
                pre_filter_count,
                guide_latent_shape,
                strength=strength,
                attention_mask=None,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})
