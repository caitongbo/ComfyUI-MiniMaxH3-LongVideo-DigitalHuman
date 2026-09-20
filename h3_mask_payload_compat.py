"""Lazy AV denoise-mask payload compatibility for latent-tail continuation.

The digital-human context node reaches this wrapper through
``ensure_existing_video_compat()``.

Native ComfyUI support always wins. Detection intentionally avoids depending on
``inspect.getsource()`` so packaged/compiled builds and harmless upstream
refactors still self-retire the compatibility wrapper when PR #15375-equivalent
MiniMaxH3 payload support is present.
"""

from __future__ import annotations

import functools
import inspect
import logging
import types

import torch

import comfy.conds
import comfy.model_base as model_base
import comfy.utils as utils

_LOG = logging.getLogger("h3_digital_human.mask_payload")
_MARKER = "_h3_existing_video_av_mask_payload_compat_v2"


def _walk_wrapped(fn):
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        yield fn
        fn = getattr(fn, "__wrapped__", None)


def _is_ours(fn):
    code = getattr(fn, "__code__", None)
    return bool(
        getattr(fn, _MARKER, False)
        and code is not None
        and code.co_name == "wrapper"
        and code.co_filename == __file__
    )


def _signature_has(fn, *names):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in names)


def _code_strings(code):
    """Yield strings/names from a function and any nested code objects."""
    if not isinstance(code, types.CodeType):
        return
    yield from code.co_names
    def walk_const(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, types.CodeType):
            yield from _code_strings(value)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                yield from walk_const(item)

    for value in code.co_consts:
        yield from walk_const(value)


def _function_mentions_native_payload(fn):
    """Structural probe for native video+audio mask conditions.

    Bytecode constants are available even when source files are stripped, and
    unlike the old source-text check this survives formatting/helper movement
    inside ``extra_conds``.
    """
    for item in _walk_wrapped(fn):
        if _is_ours(item):
            continue
        strings = set(_code_strings(getattr(item, "__code__", None)) or ())
        if "denoise_mask" in strings and "audio_denoise_mask" in strings:
            return True
    return False


def _function_calls_native_helper(fn):
    """Recognize the merged PR #15375 helper path in current ComfyUI.

    Current H3 extra_conds no longer needs to spell both output keys itself;
    it delegates mask extraction to _denoise_mask_conds().  Looking for that
    call plus the denoise_mask input lets this compatibility layer retire on
    the native helper architecture without depending on source text.
    """
    for item in _walk_wrapped(fn):
        if _is_ours(item):
            continue
        strings = set(_code_strings(getattr(item, "__code__", None)) or ())
        if "denoise_mask" in strings and "_denoise_mask_conds" in strings:
            return True
    return False


def _native_h3_mask_hooks(cls, fn):
    """Recognize native PR #15375 model_base implementations, old and new."""
    if cls is None:
        return False

    # Current merged architecture (Aug-15+): the mask work lives in four
    # MiniMaxH3 helpers and scale_latent_inpaint consumes x+denoise_mask.
    scale = cls.__dict__.get("scale_latent_inpaint")
    merged_helpers = all(callable(cls.__dict__.get(name)) for name in (
        "_pool_masks_to_token_grid",
        "_token_grid_masks",
        "_denoise_mask_values",
        "_denoise_mask_conds",
    ))
    merged_path = bool(
        merged_helpers
        and _function_calls_native_helper(fn)
        and callable(scale)
        and _signature_has(scale, "x", "denoise_mask")
    )
    if merged_path:
        return True

    # Older native architecture retained for backwards compatibility with
    # ComfyUI builds from before the helper refactor.  Our own compatibility
    # layer deliberately does not install process_timestep, so this remains a
    # useful native-only fingerprint for those versions.
    process_step = cls.__dict__.get("process_timestep")
    process_mask = cls.__dict__.get("process_denoise_mask")
    return bool(
        callable(process_step)
        and _signature_has(process_step, "denoise_mask", "audio_denoise_mask")
        and callable(process_mask)
        and callable(scale)
    )


def _native_av_mask_payload(cls, fn):
    # Direct evidence takes priority. The native-hook fingerprint is the
    # self-retirement path for upstream refactors that move payload extraction
    # into a helper and leave no literal output-key strings in extra_conds.
    return bool(
        fn
        and (
            _function_mentions_native_payload(fn)
            or _native_h3_mask_hooks(cls, fn)
        )
    )


def capability_status():
    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    direct = bool(fn and _function_mentions_native_payload(fn))
    native_hooks = _native_h3_mask_hooks(cls, fn)
    native = bool(fn and (direct or native_hooks))
    return {
        "available": fn is not None,
        "native_av_mask_payload": native,
        "native_payload_direct": direct,
        "native_h3_mask_hooks": native_hooks,
        "wrapper_present": bool(fn and any(_is_ours(x) for x in _walk_wrapped(fn))),
    }


def _add_av_mask_conditions(out, kwargs):
    if not isinstance(out, dict):
        return
    have_video = "denoise_mask" in out
    have_audio = "audio_denoise_mask" in out
    if have_video and have_audio:
        return

    denoise_mask = kwargs.get("denoise_mask")
    latent_shapes = kwargs.get("latent_shapes")
    if denoise_mask is None or latent_shapes is None or len(latent_shapes) < 2:
        return

    masks = utils.unpack_latents(denoise_mask, latent_shapes)
    if len(masks) < 2:
        return
    if not have_video and torch.amin(masks[0]).item() < 1.0 - 1e-3:
        out["denoise_mask"] = comfy.conds.CONDRegular(masks[0][:, :1].clone())
    if not have_audio and torch.amin(masks[1]).item() < 1.0 - 1e-3:
        out["audio_denoise_mask"] = comfy.conds.CONDRegular(masks[1][:, :1].clone())


def _make_wrapper(base):
    @functools.wraps(base, updated=())
    def wrapper(self, **kwargs):
        out = base(self, **kwargs)
        _add_av_mask_conditions(out, kwargs)
        return out
    setattr(wrapper, _MARKER, True)
    return wrapper


def ensure_av_mask_payload_compat():
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        raise RuntimeError("h3_digital_human: MiniMaxH3.extra_conds not found")
    current = cls.extra_conds
    if _native_av_mask_payload(cls, current):
        return True
    if _is_ours(current):
        return True
    cls.extra_conds = _make_wrapper(current)
    _LOG.info("h3_digital_human: PR #15375 AV-mask payload compatibility enabled")
    return True
