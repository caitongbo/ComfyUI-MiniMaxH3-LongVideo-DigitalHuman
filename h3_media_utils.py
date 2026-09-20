"""Media helpers used by the checkpoint-only digital-human assembler."""

from __future__ import annotations

import gc

import torch


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def pixel_frames(latent_steps):
    """Return the decoded picture-frame count for an H3 video latent."""
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(int(latent_steps)))


def decode_h3_video_cpu(vae, latent):
    """Decode one H3 video latent into contiguous CPU RGB frames."""
    images = vae.decode(latent)
    if getattr(images, "ndim", 0) == 5 and int(images.shape[0]) == 1:
        images = images[0]
    if getattr(images, "ndim", 0) != 4:
        raise ValueError(
            "h3_video_decode: video VAE decode returned %s; expected 4-D frames"
            % (tuple(getattr(images, "shape", ())),)
        )
    if int(images.shape[-1]) in (3, 4):
        images = images[..., :3]
    elif int(images.shape[1]) in (3, 4):
        images = images.movedim(1, -1)[..., :3]
    else:
        raise ValueError(
            "h3_video_decode: cannot infer RGB channel axis from %s"
            % (tuple(images.shape),)
        )
    return images.detach().to(device="cpu", dtype=torch.float32).contiguous()


def release_decode_memory():
    """Release CPU objects and ask ComfyUI to empty reclaimable device cache."""
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception:
        pass
