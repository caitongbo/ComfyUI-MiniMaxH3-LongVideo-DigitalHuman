"""Schedule-matched video-prefix masks for MiniMax H3 continuation.

Adapted for this GPL-3.0 project from TimelineDirector's Drift-Control AV,
which in turn credits ethanfel/ComfyUI-MiniMaxH3-Contex-Loop.  The saved
predecessor checkpoint is never changed: only the disposable prefix in the
active sampling run is progressively re-noised.
"""

from __future__ import annotations

import math

import torch


TAPER_STEPS = 4
_WRAPPER_KEY = "h3_digital_human_drift_control_av"


def _schedule_values(sigmas):
    if torch.is_tensor(sigmas):
        values = sigmas.detach().float().reshape(-1).cpu()
    else:
        values = sigmas or ()
    normalized = []
    for value in values:
        number = float(value)
        if math.isfinite(number) and number >= 0.0:
            normalized.append(number)
    return tuple(sorted(set(normalized), reverse=True))


def next_schedule_sigma(current_sigma, sigmas):
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    tolerance = max(1e-7, abs(current) * 1e-6)
    for candidate in _schedule_values(sigmas):
        if candidate < current - tolerance:
            return candidate
    return 0.0


def matched_noise_ratio(current_sigma, sigmas):
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    return max(0.0, min(1.0, next_schedule_sigma(current, sigmas) / current))


def temporal_prefix_weights(prefix_steps, taper_steps=TAPER_STEPS):
    count = int(prefix_steps)
    taper = min(int(taper_steps), count)
    if count < 1 or taper < 1:
        raise ValueError("h3_drift_control: prefix and taper must be positive")
    weights = [1.0] * (count - taper)
    weights.extend(
        float(taper - offset - 1) / float(taper)
        for offset in range(taper)
    )
    return tuple(weights)


def apply_dynamic_prefix_mask(packed_mask, video_shape, ratio, prefix_steps):
    if not torch.is_tensor(packed_mask) or packed_mask.ndim != 3:
        raise ValueError("h3_drift_control: expected packed mask [B,1,N]")
    shape = tuple(int(value) for value in video_shape)
    if len(shape) != 5 or shape[0] != int(packed_mask.shape[0]):
        raise ValueError("h3_drift_control: invalid H3 video latent shape")
    prefix_steps = int(prefix_steps)
    if prefix_steps < 1 or prefix_steps >= shape[2]:
        raise ValueError("h3_drift_control: prefix must leave new video tokens")
    video_elements = math.prod(shape[1:])
    if int(packed_mask.shape[-1]) < video_elements:
        raise ValueError("h3_drift_control: packed mask is shorter than video")

    output = packed_mask.clone()
    video_mask = output[..., :video_elements].reshape(shape)
    weights = torch.tensor(
        temporal_prefix_weights(prefix_steps),
        device=video_mask.device,
        dtype=video_mask.dtype,
    ).mul_(float(max(0.0, min(1.0, ratio))))
    video_mask[:, :, :prefix_steps] = weights.view(
        1, 1, prefix_steps, 1, 1
    )
    # Match H3's token-mask quantization so sampler blending and the model's
    # per-row timestep labels see exactly the same mask.
    h3_video_mask = torch.ceil(video_mask[:, :1].float() * 256.0) / 256.0
    return output, h3_video_mask


def _streams(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif hasattr(samples, "tensors"):
        parts = list(samples.tensors)
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        parts = []
    if len(parts) < 2 or not torch.is_tensor(parts[0]) or not torch.is_tensor(parts[1]):
        raise ValueError("h3_drift_control: expected a joint H3 AV latent")
    return parts[0], parts[1]


class _MaskState:
    def __init__(self, video_shape, audio_shape, sigmas, prefix_steps):
        self.video_shape = tuple(int(value) for value in video_shape)
        self.audio_shape = tuple(int(value) for value in audio_shape)
        self.sigmas = _schedule_values(sigmas)
        self.prefix_steps = int(prefix_steps)
        self.video_mask = None
        self.audio_mask = None
        self.current_packed_mask = None
        self.current_video_mask = None
        self.current_audio_mask = None

    def _update(self, sigma, packed_mask):
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        output, self.video_mask = apply_dynamic_prefix_mask(
            packed_mask,
            self.video_shape,
            matched_noise_ratio(current, self.sigmas),
            self.prefix_steps,
        )
        video_elements = math.prod(self.video_shape[1:])
        audio = output[..., video_elements:]
        if int(audio.numel()) != math.prod(self.audio_shape):
            raise ValueError("h3_drift_control: packed audio-mask shape mismatch")
        self.audio_mask = audio.reshape(self.audio_shape)[:, :1].clone()
        self.current_packed_mask = output
        self.current_video_mask = self.video_mask
        self.current_audio_mask = self.audio_mask
        return output

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        if not self.sigmas:
            self.sigmas = _schedule_values((extra_options or {}).get("sigmas", ()))
        return self._update(sigma, denoise_mask)

    def apply_model_wrapper(self, executor, *args, **kwargs):
        if self.video_mask is not None:
            kwargs["denoise_mask"] = self.video_mask
        if self.audio_mask is not None:
            kwargs["audio_denoise_mask"] = self.audio_mask
        return executor(*args, **kwargs)


def install_drift_control(model, latent, sigmas, prefix_steps):
    if model is None or not callable(getattr(model, "clone", None)):
        raise ValueError("h3_drift_control: MODEL input cannot be cloned")
    video, audio = _streams(latent)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("h3_drift_control: invalid H3 AV latent dimensions")
    prefix_steps = int(prefix_steps)
    if prefix_steps < 1 or prefix_steps >= int(video.shape[2]):
        raise ValueError("h3_drift_control: prefix must leave new video tokens")

    patched = model.clone()
    options = getattr(patched, "model_options", None)
    if not isinstance(options, dict):
        raise ValueError("h3_drift_control: MODEL has no model_options")
    if callable(options.get("denoise_mask_function")):
        raise ValueError(
            "h3_drift_control: another dynamic denoise-mask patch is active"
        )
    if not callable(getattr(patched, "set_model_denoise_mask_function", None)):
        raise RuntimeError("h3_drift_control: update ComfyUI for dynamic masks")
    if not callable(getattr(patched, "add_wrapper_with_key", None)):
        raise RuntimeError("h3_drift_control: ComfyUI model wrappers are unavailable")

    from comfy.patcher_extension import WrappersMP

    state = _MaskState(video.shape, audio.shape, sigmas, prefix_steps)
    patched.set_model_denoise_mask_function(state.denoise_mask_function)
    patched.add_wrapper_with_key(
        WrappersMP.APPLY_MODEL, _WRAPPER_KEY, state.apply_model_wrapper
    )
    patched.model_options[_WRAPPER_KEY] = state
    return patched


class MiniMaxH3DriftControlModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "context_frames": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Must match the protected video context used for this continuation.",
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 20,
                    "tooltip": "Managed Clip 1 has no inherited prefix; drift control starts at Clip 2.",
                }),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Digital Human"
    DESCRIPTION = (
        "Schedule-match the inherited video prefix to each sampler sigma. "
        "The seam-side latent stays clean while disposable earlier context is "
        "re-noised, reducing progressive blur/contrast drift. Audio masks are unchanged."
    )

    def apply(self, model, latent, sigmas, context_frames=39, clip_index=1, enabled=True):
        if not bool(enabled) or int(clip_index) <= 1:
            return (model,)
        # Valid H3 runs use 5 + 17*k pixel frames and 2 + 5*k latent steps.
        frames = int(context_frames)
        if frames < 5 or (frames - 5) % 17:
            raise ValueError(
                "h3_drift_control: context_frames must be an H3 run: 5, 22, 39, 56, ..."
            )
        prefix_steps = 2 + 5 * ((frames - 5) // 17)
        return (install_drift_control(model, latent, sigmas, prefix_steps),)
