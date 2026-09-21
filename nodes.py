"""Public nodes for checkpoint-managed MiniMax H3 digital-human video."""

from __future__ import annotations

import logging
import os
import uuid

import folder_paths

from .checkpoint_manifest import (
    archive_current_take,
    checkpoint_path,
    checkpoint_record_usable,
    file_sha256,
    publish_checkpoint,
)
from .h3_audio_vae_compat import MiniMaxH3AudioVAECompatibility
from .h3_digital_human_audio_context import MiniMaxH3DigitalHumanAVContext
from .h3_streaming_vhs import MiniMaxH3StreamCheckpointDigitalHumanToVHS
from .h3_timeline import calculate_audio_timeline

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI normally provides safetensors.
    _st_load = _st_save = None


_LOG = logging.getLogger("h3_digital_human")


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_checkpoint_resume: expected a MiniMax H3 AV latent, got %r"
            % type(samples)
        )
    if len(parts) < 2:
        raise ValueError(
            "h3_checkpoint_resume: latent has no audio stream; connect the "
            "sampler output of a MiniMax H3 AV graph"
        )
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "h3_checkpoint_resume: invalid AV latent shapes %s / %s"
            % (tuple(video.shape), tuple(audio.shape))
        )
    return video, audio


def _save_indexed_av_checkpoint(
    latent, checkpoint_prefix, clip_index, prompt=None, unique_id=None
):
    if _st_save is None:
        raise RuntimeError(
            "h3_checkpoint_resume: safetensors is unavailable; cannot save checkpoints"
        )
    clip_index = int(clip_index)
    if clip_index < 1:
        raise ValueError("h3_checkpoint_resume: clip_index must be at least 1")

    video, audio = _streams_from_latent(latent)
    video = video.cpu().contiguous()
    audio = audio.cpu().contiguous()
    folder, filename, _counter, _subfolder, _prefix = folder_paths.get_save_image_path(
        checkpoint_prefix, folder_paths.get_output_directory()
    )
    path = os.path.join(folder, "%s_%05d.safetensors" % (filename, clip_index))
    temporary = "%s.part.%s.%s" % (path, os.getpid(), uuid.uuid4().hex)
    try:
        _st_save(
            {"video": video, "audio": audio},
            temporary,
            metadata={"format": "h3_digital_human_av"},
        )
        with open(temporary, "rb+") as saved:
            os.fsync(saved.fileno())
        archive_current_take(checkpoint_prefix, clip_index, path)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    manifest = publish_checkpoint(
        checkpoint_prefix,
        clip_index,
        path,
        tuple(video.shape),
        tuple(audio.shape),
        prompt=prompt,
        unique_id=unique_id,
    )
    _LOG.info(
        "h3_checkpoint_resume: saved Clip %d AV latent to %s (video %s, audio %s) "
        "and published %s",
        clip_index,
        path,
        tuple(video.shape),
        tuple(audio.shape),
        manifest,
    )
    return path


def _load_indexed_av_checkpoint(checkpoint_prefix, clip_index):
    if _st_load is None:
        raise RuntimeError(
            "h3_checkpoint_resume: safetensors is unavailable; cannot load checkpoints"
        )
    path = checkpoint_path(checkpoint_prefix, clip_index)
    data = _st_load(path)
    if "video" not in data or "audio" not in data:
        raise ValueError("h3_checkpoint_resume: invalid AV checkpoint")
    # Native H3 VAEDecode accesses ``latent.is_nested`` directly.  A plain
    # Python list is sufficient for our internal stream helpers but is not a
    # valid ComfyUI LATENT payload at an output socket.  Reconstruct the same
    # NestedTensor container produced by the native H3 sampler.
    import comfy.nested_tensor

    latent = {
        "samples": comfy.nested_tensor.NestedTensor(
            (data["video"], data["audio"])
        ),
        "h3_checkpoint_prefix": str(checkpoint_prefix),
        "h3_checkpoint_index": int(clip_index),
        "h3_checkpoint_sha256": file_sha256(path),
    }
    return latent, path


class MiniMaxH3CheckpointResume:
    """Lazy checkpoint gate that evaluates only missing or stale clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": (
                    "LATENT",
                    {
                        "lazy": True,
                        "tooltip": (
                            "Sampler result. A verified matching checkpoint skips "
                            "this input lazily."
                        ),
                    },
                ),
                "checkpoint_prefix": (
                    "STRING",
                    {
                        "default": "h3_digital_human/checkpoints/clip",
                        "tooltip": "Shared project prefix under ComfyUI output.",
                    },
                ),
                "clip_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "reuse_checkpoint": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse a verified matching checkpoint.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "checkpoint_path")
    FUNCTION = "resume_or_save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Digital Human"
    DESCRIPTION = (
        "Load a verified clip checkpoint or atomically publish the sampler result."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def check_lazy_status(
        self,
        checkpoint_prefix,
        clip_index,
        reuse_checkpoint=True,
        latent=None,
        prompt=None,
        unique_id=None,
    ):
        if bool(reuse_checkpoint):
            usable, reason = checkpoint_record_usable(
                checkpoint_prefix, clip_index, prompt, unique_id
            )
            if usable:
                _LOG.info(
                    "h3_checkpoint_resume: Clip %d reuses verified checkpoint",
                    int(clip_index),
                )
                return []
            _LOG.info(
                "h3_checkpoint_resume: Clip %d needs generation (%s)",
                int(clip_index),
                reason,
            )
        return [] if latent is not None else ["latent"]

    def resume_or_save(
        self,
        checkpoint_prefix,
        clip_index,
        reuse_checkpoint=True,
        latent=None,
        prompt=None,
        unique_id=None,
    ):
        if latent is None:
            usable, reason = checkpoint_record_usable(
                checkpoint_prefix, clip_index, prompt, unique_id
            )
            if not usable:
                raise RuntimeError(
                    "h3_checkpoint_resume: Clip %d cannot resume: %s"
                    % (int(clip_index), reason)
                )
            return _load_indexed_av_checkpoint(checkpoint_prefix, clip_index)

        path = _save_indexed_av_checkpoint(
            latent,
            checkpoint_prefix,
            clip_index,
            prompt=prompt,
            unique_id=unique_id,
        )
        published = dict(latent)
        published["h3_checkpoint_prefix"] = str(checkpoint_prefix)
        published["h3_checkpoint_index"] = int(clip_index)
        published["h3_checkpoint_sha256"] = file_sha256(path)
        return published, path


class MiniMaxH3AudioTimelinePlan:
    """Calculate a safe checkpoint timeline from decoded master audio."""

    MAX_CLIPS = 20

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_audio": ("AUDIO",),
                "raw_clip_frames": (
                    "INT",
                    {"default": 362, "min": 5, "max": 9999, "step": 1},
                ),
                "context_frames": (
                    "INT",
                    {"default": 39, "min": 5, "max": 9998, "step": 1},
                ),
                "clip_index": (
                    "INT",
                    {"default": 1, "min": 1, "max": cls.MAX_CLIPS, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT", "AUDIO")
    RETURN_NAMES = ("clip_count", "clip_audio")
    FUNCTION = "plan"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Digital Human"
    DESCRIPTION = (
        "Read the decoded master-audio duration and calculate the number of "
        "independently checkpointed clips required by the reusable clip pipeline."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def plan(
        self,
        master_audio,
        raw_clip_frames=362,
        context_frames=39,
        clip_index=1,
    ):
        waveform = master_audio.get("waveform") if isinstance(master_audio, dict) else None
        sample_rate = int(master_audio.get("sample_rate", 0)) if isinstance(master_audio, dict) else 0
        if getattr(waveform, "ndim", 0) != 3 or int(waveform.shape[-1]) < 1:
            raise ValueError("h3_timeline_plan: master_audio must contain waveform [B,C,L]")
        if sample_rate < 1:
            raise ValueError("h3_timeline_plan: master_audio sample_rate must be positive")

        samples = int(waveform.shape[-1])
        payload = calculate_audio_timeline(
            samples,
            sample_rate,
            raw_clip_frames,
            context_frames,
            max_clips=self.MAX_CLIPS,
        )
        count = payload["clip_count"]
        index = max(1, min(self.MAX_CLIPS, int(clip_index)))
        start_frame = (index - 1) * payload["advance_frames"]
        start_sample = round(start_frame / 24.0 * sample_rate)
        end_sample = round(
            (start_frame + payload["raw_clip_frames"]) / 24.0 * sample_rate
        )
        wanted = end_sample - start_sample
        if start_sample < samples:
            clip_waveform = waveform[..., start_sample:min(end_sample, samples)]
        else:
            clip_waveform = waveform[..., :0]
        missing = wanted - int(clip_waveform.shape[-1])
        if missing > 0:
            import torch
            clip_waveform = torch.nn.functional.pad(clip_waveform, (0, missing))
        clip_audio = {"waveform": clip_waveform, "sample_rate": sample_rate}
        _LOG.info(
            "h3_timeline_plan: %.3fs master audio -> %d clips; %d raw frames, "
            "%d context frames, %.3fs coverage",
            payload["duration_seconds"],
            count,
            payload["raw_clip_frames"],
            payload["context_frames"],
            payload["coverage_seconds"],
        )
        return {"ui": {"timeline": [payload]}, "result": (count, clip_audio)}


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3AudioVAECompatibility": MiniMaxH3AudioVAECompatibility,
    "MiniMaxH3CheckpointResume": MiniMaxH3CheckpointResume,
    "MiniMaxH3DigitalHumanAVContext": MiniMaxH3DigitalHumanAVContext,
    "MiniMaxH3StreamCheckpointDigitalHumanToVHS": MiniMaxH3StreamCheckpointDigitalHumanToVHS,
    "MiniMaxH3AudioTimelinePlan": MiniMaxH3AudioTimelinePlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AudioVAECompatibility": "H3 Digital Human — Audio VAE Compatibility",
    "MiniMaxH3CheckpointResume": "H3 Digital Human — Checkpoint Resume Gate",
    "MiniMaxH3DigitalHumanAVContext": "H3 Digital Human — Master Audio + Video Context",
    "MiniMaxH3StreamCheckpointDigitalHumanToVHS": "H3 Digital Human — Assemble Checkpoints",
    "MiniMaxH3AudioTimelinePlan": "H3 Digital Human — Audio Timeline Plan",
}
