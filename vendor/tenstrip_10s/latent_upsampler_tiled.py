"""
Vendored tiled latent upsampler adapted from TenStrip/10S-Comfy-nodes.

This helper intentionally does not register a ComfyUI node. WhatDreamsCost
exposes it through the Director-specific tiled upscale guide node instead.
"""

from __future__ import annotations

import math

import torch

try:
    import comfy.model_management as model_management
except Exception:  # noqa: BLE001 - tests and non-Comfy imports can use CPU fallbacks.
    model_management = None


class LTXVLatentUpsamplerTiled:
    """Spatially tiled drop-in replacement for LTXVLatentUpsampler."""

    def upsample_latent_tiled(
        self,
        samples,
        upscale_model,
        vae,
        tile_size=24,
        overlap=8,
        max_size_for_no_tile=32,
        rotate_for_landscape=False,
        debug=False,
    ):
        latents = samples["samples"]
        input_dtype = latents.dtype
        model_dtype = self._model_dtype(upscale_model, input_dtype)
        device = self._torch_device(latents)

        if latents.ndim != 5:
            raise ValueError(f"LTX tiled upsampler expects 5D latent samples, got shape {tuple(latents.shape)}")

        if overlap >= tile_size:
            if debug:
                print(
                    "[WDC] LTX tiled upsampler: overlap >= tile_size; "
                    f"clamping {overlap} to {tile_size - 1}."
                )
            overlap = max(1, tile_size - 1)

        _, _, _, h, w = latents.shape
        self._free_memory(upscale_model, latents, tile_size, h, w, device)

        moved_model = False
        try:
            if hasattr(upscale_model, "to"):
                upscale_model.to(device)
                moved_model = True

            latents_dev = latents.to(dtype=model_dtype, device=device)
            latents_un = self._unnormalize(vae, latents_dev)

            rotated = False
            if rotate_for_landscape and latents_un.shape[-2] > latents_un.shape[-1]:
                latents_un = latents_un.transpose(-1, -2).contiguous()
                rotated = True
                if debug:
                    print(f"[WDC] LTX tiled upsampler: rotated to {tuple(latents_un.shape)}")

            _, _, _, h, w = latents_un.shape
            should_tile = h > max_size_for_no_tile or w > max_size_for_no_tile
            if should_tile:
                upsampled = self._upsample_tiled(latents_un, upscale_model, tile_size, overlap, debug)
            else:
                if debug:
                    print("[WDC] LTX tiled upsampler: using single-pass upscale.")
                upsampled = upscale_model(latents_un)

            if rotated:
                upsampled = upsampled.transpose(-1, -2).contiguous()

            upsampled = self._normalize(vae, upsampled)
        finally:
            if moved_model and hasattr(upscale_model, "cpu"):
                upscale_model.cpu()

        output = samples.copy()
        output["samples"] = upsampled.to(dtype=input_dtype, device=self._intermediate_device())
        output.pop("noise_mask", None)
        return (output,)

    def _upsample_tiled(self, latents, upscale_model, tile_size, overlap, debug):
        device = latents.device
        dtype = latents.dtype
        batch, channels, frames, height, width = latents.shape
        h_starts = self._compute_tile_starts(height, tile_size, overlap)
        w_starts = self._compute_tile_starts(width, tile_size, overlap)

        first_h_end = min(h_starts[0] + tile_size, height)
        first_w_end = min(w_starts[0] + tile_size, width)
        first_tile_in = latents[:, :, :, h_starts[0]:first_h_end, w_starts[0]:first_w_end].contiguous()
        first_tile_out = upscale_model(first_tile_in)

        scale_h = first_tile_out.shape[3] / first_tile_in.shape[3]
        scale_w = first_tile_out.shape[4] / first_tile_in.shape[4]
        scale = (scale_h + scale_w) / 2.0
        if abs(scale_h - scale_w) > 0.01:
            print(
                "[WDC] LTX tiled upsampler: non-uniform scale detected "
                f"({scale_h:.3f} vs {scale_w:.3f}); using {scale:.3f}."
            )

        out_h = int(round(height * scale))
        out_w = int(round(width * scale))
        if debug:
            print(
                "[WDC] LTX tiled upsampler: "
                f"{height}x{width} -> {out_h}x{out_w}, "
                f"tile_size={tile_size}, overlap={overlap}, "
                f"tiles={len(h_starts) * len(w_starts)}"
            )

        output = torch.zeros((batch, channels, frames, out_h, out_w), dtype=torch.float32, device=device)
        weights = torch.zeros((1, 1, 1, out_h, out_w), dtype=torch.float32, device=device)
        single_h = len(h_starts) == 1
        single_w = len(w_starts) == 1

        for h_idx, h_start in enumerate(h_starts):
            h_end = min(h_start + tile_size, height)
            for w_idx, w_start in enumerate(w_starts):
                w_end = min(w_start + tile_size, width)
                if h_idx == 0 and w_idx == 0:
                    tile_out = first_tile_out
                else:
                    tile_in = latents[:, :, :, h_start:h_end, w_start:w_end].contiguous()
                    tile_out = upscale_model(tile_in)

                out_h_start = int(round(h_start * scale))
                out_w_start = int(round(w_start * scale))
                out_h_end = min(out_h_start + tile_out.shape[3], out_h)
                out_w_end = min(out_w_start + tile_out.shape[4], out_w)
                actual_h = out_h_end - out_h_start
                actual_w = out_w_end - out_w_start
                if actual_h <= 0 or actual_w <= 0:
                    continue

                fade_top = 0 if single_h else int(round(self._actual_overlap_prev(h_starts, h_idx, tile_size, height) * scale))
                fade_bottom = 0 if single_h else int(round(self._actual_overlap_next(h_starts, h_idx, tile_size, height) * scale))
                fade_left = 0 if single_w else int(round(self._actual_overlap_prev(w_starts, w_idx, tile_size, width) * scale))
                fade_right = 0 if single_w else int(round(self._actual_overlap_next(w_starts, w_idx, tile_size, width) * scale))

                window = self._make_window_2d(
                    actual_h,
                    actual_w,
                    min(fade_top, actual_h),
                    min(fade_bottom, actual_h),
                    min(fade_left, actual_w),
                    min(fade_right, actual_w),
                    device,
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)

                tile_crop = tile_out[:, :, :, :actual_h, :actual_w]
                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += tile_crop.float() * window
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += window

        if debug:
            print(
                "[WDC] LTX tiled upsampler: weight accumulator "
                f"min={weights.min().item():.4f} max={weights.max().item():.4f}"
            )

        return (output / (weights + 1e-8)).to(dtype=dtype)

    @staticmethod
    def _compute_tile_starts(total_size, tile_size, overlap):
        if total_size <= tile_size:
            return [0]
        starts = []
        stride = tile_size - overlap
        pos = 0
        while pos + tile_size < total_size:
            starts.append(pos)
            pos += stride
        last_start = total_size - tile_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _actual_overlap_prev(starts, idx, tile_size, total):
        if idx <= 0:
            return 0
        prev_end = min(starts[idx - 1] + tile_size, total)
        return max(0, prev_end - starts[idx])

    @staticmethod
    def _actual_overlap_next(starts, idx, tile_size, total):
        if idx >= len(starts) - 1:
            return 0
        this_end = min(starts[idx] + tile_size, total)
        return max(0, this_end - starts[idx + 1])

    @staticmethod
    def _make_window_1d(size, fade_left_size, fade_right_size, device):
        win = torch.ones(size, dtype=torch.float32, device=device)
        if fade_left_size > 0:
            fade = min(fade_left_size, size)
            idx = torch.arange(fade, dtype=torch.float32, device=device)
            win[:fade] = 0.5 * (1.0 - torch.cos(math.pi * idx / fade))
        if fade_right_size > 0:
            fade = min(fade_right_size, size)
            idx = torch.arange(fade, dtype=torch.float32, device=device)
            win[size - fade:] = 0.5 * (1.0 + torch.cos(math.pi * idx / fade))
        return win

    @classmethod
    def _make_window_2d(cls, height, width, fade_top, fade_bottom, fade_left, fade_right, device):
        win_h = cls._make_window_1d(height, fade_top, fade_bottom, device)
        win_w = cls._make_window_1d(width, fade_left, fade_right, device)
        return win_h.unsqueeze(1) * win_w.unsqueeze(0)

    @staticmethod
    def _model_dtype(upscale_model, fallback):
        try:
            return next(upscale_model.parameters()).dtype
        except Exception:  # noqa: BLE001 - non-module test doubles may not expose parameters.
            return fallback

    @staticmethod
    def _unnormalize(vae, latents):
        stats = getattr(getattr(vae, "first_stage_model", None), "per_channel_statistics", None)
        if stats is not None and hasattr(stats, "un_normalize"):
            return stats.un_normalize(latents)
        return latents

    @staticmethod
    def _normalize(vae, latents):
        stats = getattr(getattr(vae, "first_stage_model", None), "per_channel_statistics", None)
        if stats is not None and hasattr(stats, "normalize"):
            return stats.normalize(latents)
        return latents

    @staticmethod
    def _torch_device(latents):
        if model_management is not None and hasattr(model_management, "get_torch_device"):
            return model_management.get_torch_device()
        return latents.device

    @staticmethod
    def _intermediate_device():
        if model_management is not None and hasattr(model_management, "intermediate_device"):
            return model_management.intermediate_device()
        return "cpu"

    @staticmethod
    def _free_memory(upscale_model, latents, tile_size, height, width, device):
        if model_management is None or not hasattr(model_management, "free_memory"):
            return
        try:
            model_size = model_management.module_size(upscale_model)
        except Exception:  # noqa: BLE001 - keep memory hints best-effort.
            model_size = 0
        batch, channels, frames, _, _ = latents.shape
        tile_volume = batch * channels * frames * (tile_size * 2) ** 2
        output_volume = batch * channels * frames * (height * 2) * (width * 2)
        model_management.free_memory(model_size + tile_volume * 3000.0 + output_volume * 4.0, device)
