from copy import deepcopy

from comfy_api.latest import io


IdentityAnchor = io.Custom("LTX_IDENTITY_ANCHOR")
GuideData = io.Custom("GUIDE_DATA")

_TEN_S_MODULE_CACHE = {}


def _load_10s_class(filename, class_name):
    cache_key = class_name
    if cache_key in _TEN_S_MODULE_CACHE:
        return _TEN_S_MODULE_CACHE[cache_key]

    try:
        if class_name == "LTXLatentAnchorAware":
            try:
                from .vendor.tenstrip_10s.latent_anchor_aware import LTXLatentAnchorAware
            except ImportError:
                from vendor.tenstrip_10s.latent_anchor_aware import LTXLatentAnchorAware

            klass = LTXLatentAnchorAware
        elif class_name == "LTXFaceAttentionAnchor":
            try:
                from .vendor.tenstrip_10s.face_anchor import LTXFaceAttentionAnchor
            except ImportError:
                from vendor.tenstrip_10s.face_anchor import LTXFaceAttentionAnchor

            klass = LTXFaceAttentionAnchor
        else:
            klass = None
    except Exception as exc:  # noqa: BLE001 - keep workflows from failing at graph construction.
        print(
            f"[WDC] LTX Identity Anchor: bundled {class_name} could not be loaded "
            f"from {filename}: {type(exc).__name__}: {exc}"
        )
        klass = None

    _TEN_S_MODULE_CACHE[cache_key] = klass
    return klass


def _first_guide_image(guide_data):
    if not isinstance(guide_data, dict):
        return None
    images = guide_data.get("images") or []
    return images[0] if images else None


def select_director_reference_image(guide_data, reference_label="image1"):
    if not isinstance(guide_data, dict):
        raise ValueError("LTX Director reference image selector needs guide_data from an LTX Director node.")

    references = guide_data.get("reference_images") or []
    if not references:
        raise ValueError("LTX Director guide_data does not contain any character reference images.")

    label = str(reference_label or "").strip().lower()
    if not label:
        entry = references[0]
    else:
        entry = next(
            (
                ref for ref in references
                if str(ref.get("label") or "").strip().lower() == label
                or str(ref.get("id") or "").strip().lower() == label
            ),
            None,
        )
        if entry is None:
            available = sorted(
                {
                    str(ref.get("label") or ref.get("id") or "").strip()
                    for ref in references
                    if str(ref.get("label") or ref.get("id") or "").strip()
                }
            )
            suffix = f" Available references: {', '.join(available)}." if available else ""
            raise ValueError(f"LTX Director reference image '{reference_label}' was not found.{suffix}")

    image = entry.get("image")
    if image is None:
        raise ValueError(f"LTX Director reference image '{entry.get('label') or reference_label}' has no loaded image tensor.")
    return image


def _scaled_anchor(anchor, strength_scale):
    if not isinstance(anchor, dict):
        return anchor
    scaled = deepcopy(anchor)
    if "strength" in scaled:
        scaled["strength"] = float(scaled["strength"]) * float(strength_scale)
    return scaled


def _ordered_anchors(identity_anchor):
    if identity_anchor is None:
        return []
    if isinstance(identity_anchor, dict) and identity_anchor.get("kind") == "combined":
        anchors = identity_anchor.get("anchors", [])
        if identity_anchor.get("scale_strengths", True):
            scale = identity_anchor.get("strength_scale", 0.75)
            anchors = [_scaled_anchor(anchor, scale) for anchor in anchors]
    else:
        anchors = [identity_anchor]

    anchors = [anchor for anchor in anchors if isinstance(anchor, dict)]
    order = {"latent_aware": 0, "face": 1}
    return sorted(anchors, key=lambda anchor: order.get(anchor.get("kind"), 99))


def _apply_latent_aware(model, anchor, sigmas=None, vae=None, guide_data=None):
    klass = _load_10s_class("latent_anchor_aware.py", "LTXLatentAnchorAware")
    if klass is None:
        print("[WDC] LTX Identity Anchor: bundled LTXLatentAnchorAware unavailable; bypassing.")
        return model

    energy_source = anchor.get("energy_source", "auto")
    reference_image = anchor.get("reference_image")
    energy_latent = anchor.get("energy_latent")

    if energy_source == "none":
        reference_image = None
        energy_latent = None
    elif energy_source == "reference_image":
        energy_latent = None
        if reference_image is None:
            print("[WDC] LTX Identity Anchor: reference_image selected but not connected.")
    elif energy_source == "energy_latent":
        reference_image = None
        if energy_latent is None:
            print("[WDC] LTX Identity Anchor: energy_latent selected but not connected.")
    elif energy_source == "first_guide_image":
        reference_image = _first_guide_image(guide_data)
        energy_latent = None
        if reference_image is None:
            print("[WDC] LTX Identity Anchor: first guide image selected but guide_data is empty.")
    else:
        if reference_image is not None:
            energy_latent = None
        elif energy_latent is not None:
            reference_image = None
        else:
            reference_image = _first_guide_image(guide_data)

    if reference_image is not None and vae is None:
        print(
            "[WDC] LTX Identity Anchor: reference image energy needs a VAE; "
            "10S will continue with energy modulation disabled."
        )

    return klass().patch(
        model,
        reference_image=reference_image,
        vae=vae,
        energy_latent=energy_latent,
        sigmas=sigmas,
        strength=anchor.get("strength", 0.10),
        cache_at_step=anchor.get("cache_at_step", 6),
        similarity_threshold=anchor.get("similarity_threshold", 0.50),
        decay_with_distance=anchor.get("decay_with_distance", 0.0),
        energy_threshold=anchor.get("energy_threshold", 0.30),
        bypass=anchor.get("bypass", False),
        debug=anchor.get("debug", False),
        advanced_mode=anchor.get("advanced_mode", False),
        cache_mode=anchor.get("cache_mode", "schedule"),
        forwards_per_step=anchor.get("forwards_per_step", 1),
        cache_warmup=anchor.get("cache_warmup", 144),
        anchor_frame=anchor.get("anchor_frame", 0),
        depth_curve=anchor.get("depth_curve", "flat"),
        block_index_filter=anchor.get("block_index_filter", ""),
    )[0]


def _apply_face(model, anchor):
    klass = _load_10s_class("face_anchor.py", "LTXFaceAttentionAnchor")
    if klass is None:
        print("[WDC] LTX Identity Anchor: bundled LTXFaceAttentionAnchor unavailable; bypassing.")
        return model

    return klass().patch(
        model,
        face_bbox_norm=anchor.get("face_bbox_norm", "0.35,0.10,0.65,0.50"),
        strength=anchor.get("strength", 0.10),
        inject_mode=anchor.get("inject_mode", "tracked"),
        anchor_frame=anchor.get("anchor_frame", 0),
        anchor_upsample=anchor.get("anchor_upsample", 2),
        track_threshold=anchor.get("track_threshold", 0.50),
        face_threshold=anchor.get("face_threshold", 0.30),
        identity_threshold=anchor.get("identity_threshold", 0.75),
        depth_curve=anchor.get("depth_curve", "flat"),
        spatial_prior=anchor.get("spatial_prior", 0.50),
        block_index_filter=anchor.get("block_index_filter", ""),
        bypass=anchor.get("bypass", False),
        debug=anchor.get("debug", False),
    )[0]


def apply_identity_anchor(model, identity_anchor=None, sigmas=None, vae=None, guide_data=None):
    patched = model
    for anchor in _ordered_anchors(identity_anchor):
        kind = anchor.get("kind")
        if kind == "off" or anchor.get("bypass_all", False):
            continue
        if kind == "latent_aware":
            patched = _apply_latent_aware(patched, anchor, sigmas=sigmas, vae=vae, guide_data=guide_data)
        elif kind == "face":
            patched = _apply_face(patched, anchor)
        else:
            print(f"[WDC] LTX Identity Anchor: unknown anchor kind '{kind}'; bypassing.")
    return patched


class LTXIdentityAnchorLatentAware(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXIdentityAnchorLatentAware",
            display_name="LTX Identity Anchor: Latent Aware",
            category="LTXVCustom/Identity",
            description="Configures optional 10S latent-aware identity anchoring for LTX Director.",
            inputs=[
                io.Combo.Input(
                    "energy_source",
                    options=["auto", "none", "reference_image", "energy_latent", "first_guide_image"],
                    default="auto",
                    tooltip="Spatial energy source. Auto prefers reference image, then energy latent, then first Director guide image.",
                ),
                io.Image.Input("reference_image", optional=True, tooltip="Optional image used only for spatial energy weighting."),
                io.Latent.Input("energy_latent", optional=True, tooltip="Optional latent used only for spatial energy weighting."),
                io.Float.Input("strength", default=0.10, min=0.0, max=5.0, step=0.01),
                io.Int.Input("cache_at_step", default=6, min=0, max=100, step=1),
                io.Float.Input("similarity_threshold", default=0.50, min=0.0, max=1.0, step=0.01),
                io.Float.Input("decay_with_distance", default=0.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("energy_threshold", default=0.30, min=0.0, max=1.0, step=0.05),
                io.Int.Input("anchor_frame", default=0, min=0, max=256, step=1, optional=True),
                io.Boolean.Input("advanced_mode", default=False, optional=True),
                io.Combo.Input(
                    "cache_mode",
                    options=["schedule", "live_extraction", "manual_calls"],
                    default="schedule",
                    optional=True,
                ),
                io.Int.Input("forwards_per_step", default=1, min=1, max=8, step=1, optional=True),
                io.Int.Input("cache_warmup", default=144, min=0, max=5000, step=1, optional=True),
                io.Combo.Input(
                    "depth_curve",
                    options=["flat", "ramp_up", "ramp_down", "late_focus", "middle"],
                    default="flat",
                    optional=True,
                ),
                io.String.Input("block_index_filter", default="", optional=True),
                io.Boolean.Input("bypass", default=False, optional=True),
                io.Boolean.Input("debug", default=False, optional=True),
            ],
            outputs=[
                IdentityAnchor.Output(display_name="identity_anchor"),
            ],
        )

    @classmethod
    def execute(
        cls,
        energy_source="auto",
        reference_image=None,
        energy_latent=None,
        strength=0.10,
        cache_at_step=6,
        similarity_threshold=0.50,
        decay_with_distance=0.0,
        energy_threshold=0.30,
        anchor_frame=0,
        advanced_mode=False,
        cache_mode="schedule",
        forwards_per_step=1,
        cache_warmup=144,
        depth_curve="flat",
        block_index_filter="",
        bypass=False,
        debug=False,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            {
                "kind": "latent_aware",
                "energy_source": energy_source,
                "reference_image": reference_image,
                "energy_latent": energy_latent,
                "strength": strength,
                "cache_at_step": cache_at_step,
                "similarity_threshold": similarity_threshold,
                "decay_with_distance": decay_with_distance,
                "energy_threshold": energy_threshold,
                "anchor_frame": anchor_frame,
                "advanced_mode": advanced_mode,
                "cache_mode": cache_mode,
                "forwards_per_step": forwards_per_step,
                "cache_warmup": cache_warmup,
                "depth_curve": depth_curve,
                "block_index_filter": block_index_filter,
                "bypass": bypass,
                "debug": debug,
            }
        )


class LTXIdentityAnchorFace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXIdentityAnchorFace",
            display_name="LTX Identity Anchor: Face",
            category="LTXVCustom/Identity",
            description="Configures optional 10S face-region identity anchoring for LTX Director.",
            inputs=[
                io.String.Input("face_bbox_norm", default="0.35,0.10,0.65,0.50"),
                io.Float.Input("strength", default=0.10, min=0.0, max=5.0, step=0.01),
                io.Combo.Input("inject_mode", options=["tracked", "tracked_correction"], default="tracked"),
                io.Int.Input("anchor_frame", default=0, min=0, max=256, step=1),
                io.Int.Input("anchor_upsample", default=2, min=1, max=4, step=1),
                io.Float.Input("track_threshold", default=0.50, min=0.0, max=1.0, step=0.01),
                io.Float.Input("face_threshold", default=0.30, min=0.0, max=1.0, step=0.01),
                io.Float.Input("identity_threshold", default=0.75, min=0.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "depth_curve",
                    options=["flat", "ramp_up", "ramp_down", "late_focus", "middle"],
                    default="flat",
                ),
                io.Float.Input("spatial_prior", default=0.50, min=0.0, max=1.0, step=0.05),
                io.String.Input("block_index_filter", default="", optional=True),
                io.Boolean.Input("bypass", default=False, optional=True),
                io.Boolean.Input("debug", default=False, optional=True),
            ],
            outputs=[
                IdentityAnchor.Output(display_name="identity_anchor"),
            ],
        )

    @classmethod
    def execute(
        cls,
        face_bbox_norm="0.35,0.10,0.65,0.50",
        strength=0.10,
        inject_mode="tracked",
        anchor_frame=0,
        anchor_upsample=2,
        track_threshold=0.50,
        face_threshold=0.30,
        identity_threshold=0.75,
        depth_curve="flat",
        spatial_prior=0.50,
        block_index_filter="",
        bypass=False,
        debug=False,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            {
                "kind": "face",
                "face_bbox_norm": face_bbox_norm,
                "strength": strength,
                "inject_mode": inject_mode,
                "anchor_frame": anchor_frame,
                "anchor_upsample": anchor_upsample,
                "track_threshold": track_threshold,
                "face_threshold": face_threshold,
                "identity_threshold": identity_threshold,
                "depth_curve": depth_curve,
                "spatial_prior": spatial_prior,
                "block_index_filter": block_index_filter,
                "bypass": bypass,
                "debug": debug,
            }
        )


class LTXIdentityAnchorCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXIdentityAnchorCombine",
            display_name="LTX Identity Anchor: Combine",
            category="LTXVCustom/Identity",
            description="Combines latent-aware and face identity anchor configs for one adapter input.",
            inputs=[
                IdentityAnchor.Input("anchor_a", optional=True),
                IdentityAnchor.Input("anchor_b", optional=True),
                io.Boolean.Input(
                    "scale_strengths",
                    default=True,
                    tooltip="Reduce both strengths when combining to avoid over-constraining motion.",
                ),
                io.Float.Input("strength_scale", default=0.75, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                IdentityAnchor.Output(display_name="identity_anchor"),
            ],
        )

    @classmethod
    def execute(cls, anchor_a=None, anchor_b=None, scale_strengths=True, strength_scale=0.75) -> io.NodeOutput:
        anchors = [anchor for anchor in (anchor_a, anchor_b) if isinstance(anchor, dict)]
        return io.NodeOutput(
            {
                "kind": "combined",
                "anchors": anchors,
                "scale_strengths": scale_strengths,
                "strength_scale": strength_scale,
            }
        )


class LTXDirectorReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorReferenceImage",
            display_name="LTX Director Reference Image",
            category="LTXVCustom/Identity",
            description=(
                "Selects a character reference image from LTX Director guide_data so it can be "
                "connected to LTX Identity Anchor: Latent Aware.reference_image."
            ),
            inputs=[
                GuideData.Input("guide_data", tooltip="Guide data produced by LTX Director."),
                io.String.Input(
                    "reference_label",
                    default="image1",
                    tooltip="Director reference label or id, for example image1. Leave blank to use the first reference.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="reference_image"),
            ],
        )

    @classmethod
    def execute(cls, guide_data, reference_label="image1") -> io.NodeOutput:
        return io.NodeOutput(select_director_reference_image(guide_data, reference_label))


class LTXDirectorApplyIdentityAnchor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirectorApplyIdentityAnchor",
            display_name="LTX Director Apply Identity Anchor",
            category="LTXVCustom/Identity",
            description="Optionally applies a configured 10S identity anchor to the LTX Director model output.",
            inputs=[
                io.Model.Input("model"),
                IdentityAnchor.Input("identity_anchor", optional=True),
                GuideData.Input("guide_data", optional=True, tooltip="Optional Director guide data for first-guide-image energy."),
                io.Sigmas.Input("sigmas", optional=True, tooltip="Optional sampler sigmas for predictable latent-aware cache timing."),
                io.Vae.Input("vae", optional=True, tooltip="Optional VAE required when latent-aware uses reference image energy."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, identity_anchor=None, guide_data=None, sigmas=None, vae=None) -> io.NodeOutput:
        if identity_anchor is None:
            return io.NodeOutput(model)
        return io.NodeOutput(
            apply_identity_anchor(
                model,
                identity_anchor=identity_anchor,
                sigmas=sigmas,
                vae=vae,
                guide_data=guide_data,
            )
        )


NODE_CLASS_MAPPINGS = {
    "LTXIdentityAnchorLatentAware": LTXIdentityAnchorLatentAware,
    "LTXIdentityAnchorFace": LTXIdentityAnchorFace,
    "LTXIdentityAnchorCombine": LTXIdentityAnchorCombine,
    "LTXDirectorReferenceImage": LTXDirectorReferenceImage,
    "LTXDirectorApplyIdentityAnchor": LTXDirectorApplyIdentityAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXIdentityAnchorLatentAware": "LTX Identity Anchor: Latent Aware",
    "LTXIdentityAnchorFace": "LTX Identity Anchor: Face",
    "LTXIdentityAnchorCombine": "LTX Identity Anchor: Combine",
    "LTXDirectorReferenceImage": "LTX Director Reference Image",
    "LTXDirectorApplyIdentityAnchor": "LTX Director Apply Identity Anchor",
}
