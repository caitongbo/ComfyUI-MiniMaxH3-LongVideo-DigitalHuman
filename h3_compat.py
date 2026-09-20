"""Capability-aware AV-mask compatibility for the digital-human workflow."""

from __future__ import annotations

import logging


_LOG = logging.getLogger("h3_digital_human.compat")
_READY_LOGGED = False


def ensure_existing_video_compat():
    """Provide the masked AV path required by latent-tail continuation.

    Native ComfyUI capabilities are preferred. The compatibility modules only
    install behavior that the running ComfyUI build does not yet provide.
    """
    global _READY_LOGGED

    from .h3_mask_compat import ensure_h3_mask_compat
    from .h3_mask_payload_compat import ensure_av_mask_payload_compat
    from .h3_mask_velocity import ensure_h3_mask_velocity

    ensure_h3_mask_compat()
    ensure_av_mask_payload_compat()
    ensure_h3_mask_velocity()
    if not _READY_LOGGED:
        _READY_LOGGED = True
        _LOG.info(
            "h3_digital_human: H3 AV-mask continuation support ready; "
            "native ComfyUI capabilities are preferred automatically"
        )
    return True


def capability_report():
    """Return the two AV-mask capability reports used by this project."""
    from .h3_mask_compat import capability_status as mask_capability_status
    from .h3_mask_payload_compat import capability_status as payload_capability_status

    return {
        "mask": mask_capability_status(),
        "mask_payload": payload_capability_status(),
    }
