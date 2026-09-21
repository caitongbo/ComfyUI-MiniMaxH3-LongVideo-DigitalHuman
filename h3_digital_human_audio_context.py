"""MiniMax H3 digital-human clip prep with exact master-audio audio and live video context."""

import logging

import torch

from .h3_audio_grid import audio_grid_geometry, encode_exact_audio_grid

import comfy.nested_tensor
import comfy.utils

from .h3_timing import sample_boundary_from_seconds

try:
    from .h3_compat import ensure_existing_video_compat
except ImportError:
    ensure_existing_video_compat = lambda: True

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_digital_human.audio_context")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _require_h3_mask_support():
    # Native PR #15375 support wins, including the Aug-15+ no-preprocess-hook
    # architecture. The compatibility orchestrator validates the full path.
    ensure_existing_video_compat()


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_digital_human_audio: expected a MiniMax H3 AV latent, got %r" % type(samples)
        )
    if len(parts) < 2:
        raise ValueError("h3_digital_human_audio: expected joint H3 video+audio latent streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_digital_human_audio: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_digital_human_audio: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _video_steps_for_frames(frame_count):
    """Return the exact H3 video-latent step count for a valid pixel-frame run."""
    frame_count = int(frame_count)
    total = 0
    for steps in range(1, 100000):
        total += FRAME_PER_TOKEN[(steps - 1) % len(FRAME_PER_TOKEN)]
        if total == frame_count:
            return steps
        if total > frame_count:
            break
    raise ValueError(
        "h3_digital_human_audio: %d frames are not an exact H3 temporal run" % frame_count
    )


def _resize_images(images, width, height, crop, chunk=32):
    if int(images.shape[0]) <= chunk:
        x = images[..., :3].movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "lanczos", crop)
        return x.movedim(1, -1)
    out = []
    for start in range(0, int(images.shape[0]), chunk):
        part = images[start : start + chunk, ..., :3].movedim(-1, 1)
        part = comfy.utils.common_upscale(part, width, height, "lanczos", crop)
        out.append(part.movedim(1, -1))
    return torch.cat(out, dim=0)


def _cfr_index_map(frame_count, source_fps, device, target_fps=FPS):
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_digital_human_audio: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_digital_human_audio: source video has no frames")
    out_n = max(1, int(round(n * float(target_fps) / source_fps)))
    if out_n == n and abs(source_fps - target_fps) < 1e-6:
        return torch.arange(n, device=device, dtype=torch.long)
    i = torch.arange(out_n, device=device, dtype=torch.float64)
    t = (i + 0.5) / float(target_fps)
    src = torch.round(t * source_fps - 0.5).to(torch.long)
    return src.clamp_(0, n - 1)


def _stereo_first_batch(waveform, label):
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_digital_human_audio: %s waveform must be [B,C,L], got %s"
            % (label, tuple(getattr(waveform, "shape", ())))
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_digital_human_audio: %s has %d channels. Downmix multichannel audio to stereo before this node."
        % (label, channels)
    )


def _resample_waveform(waveform, source_sr, target_sr, label):
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            "h3_digital_human_audio: %s is %d Hz but %d Hz is required and torchaudio is unavailable"
            % (label, source_sr, target_sr)
        )
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _fit_waveform(waveform, want, label, pad=True):
    want = int(want)
    have = int(waveform.shape[-1])
    if have == want:
        return waveform
    if have > want:
        _LOG.info(
            "h3_digital_human_audio: trimming %s by %d samples to exact timeline",
            label,
            have - want,
        )
        return waveform[..., :want]
    if not pad:
        raise ValueError(
            "h3_digital_human_audio: %s is %d samples short of the video timeline" % (label, want - have)
        )
    _LOG.warning(
        "h3_digital_human_audio: %s is %d samples short; padding silence at tail",
        label,
        want - have,
    )
    return torch.nn.functional.pad(waveform, (0, want - have))


def _largest_h3_video_run(n):
    n = int(n)
    if n < 5:
        return 0
    return ((n - 5) // 17) * 17 + 5


def _snap_context_length(requested, available, target_frames):
    cap = min(int(requested), int(available), int(target_frames) - 1)
    run = _largest_h3_video_run(cap)
    if run < 5:
        raise ValueError(
            "h3_digital_human_audio: need at least 5 source frames and a target longer than the preserved prefix"
        )
    if run != int(requested):
        _LOG.warning(
            "h3_digital_human_audio: context_length %d -> exact H3 prefix %d (valid runs are 5, 22, 39, 56, ...; 39/90/141/... align AV clocks)",
            int(requested), run,
        )
    return run


class MiniMaxH3DigitalHumanAVContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target AV latent from MiniMax H3 Reference to Video."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE. Used to encode the master-audio slice into the target audio latent."
                }),
                "master_audio": ("AUDIO", {
                    "tooltip": "Full master audio or master vocal audio. The node writes the exact requested slice into the H3 latent and protects it from denoising."
                }),
                "checkpoint_prefix": ("STRING", {
                    "default": "h3_digital_human/checkpoints/clip",
                    "tooltip": "Shared checkpoint project prefix. Clip N loads the verified active checkpoint for Clip N-1."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 20,
                    "tooltip": "Managed reusable-pipeline index. Do not edit during a managed run."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Optional preserved visual prefix from the previous clip. Uses exact H3 runs 5/22/39/56/...; 0 disables visual prefixing entirely."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 24.0, "max": 24.0, "step": 0.001,
                    "tooltip": "FPS of source_frames when visual context is connected."
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "AUDIO")
    RETURN_NAMES = ("latent", "trim_frames", "clip_audio")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Digital Human"
    DESCRIPTION = (
        "Prepare one managed Clip: load its verified predecessor checkpoint, copy the exact latent tail, and write the frame-derived master-audio slice."
    )

    def prepare(
        self,
        latent,
        audio_vae,
        master_audio,
        clip_start_seconds=0.0,
        context_length=39,
        source_fps=24.0,
        crop="disabled",
        vae=None,
        source_frames=None,
        source_latent=None,
        checkpoint_prefix="",
        clip_index=None,
    ):
        _require_h3_mask_support()

        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError("h3_digital_human_audio: batch size 1 only")

        target_frames = _pixel_frames(int(target_video.shape[2]))

        # The reusable workflow submits this same graph once per clip.  Resolve
        # the active predecessor from the indexed manifest instead of keeping
        # twenty sampler chains in one static ComfyUI graph.  Each submission
        # still remains an independent top-level prompt and therefore retains
        # crash-safe checkpoint/resume behaviour.
        managed_index = int(clip_index) if clip_index is not None else None
        managed_prefix = str(checkpoint_prefix or "").strip()
        if managed_prefix:
            if managed_index is None or managed_index < 1:
                raise ValueError("h3_digital_human_audio: managed clip_index must be at least 1")
            if source_latent is not None or source_frames is not None:
                raise ValueError(
                    "h3_digital_human_audio: managed checkpoint continuation cannot be combined with source_latent/source_frames"
                )
            if managed_index == 1:
                clip_start_seconds = 0.0
                context_length = 0
            else:
                from .checkpoint_manifest import checkpoint_record_usable
                from .nodes import _load_indexed_av_checkpoint

                predecessor = managed_index - 1
                usable, reason = checkpoint_record_usable(
                    managed_prefix, predecessor, verify_hash=True
                )
                if not usable:
                    raise RuntimeError(
                        "h3_digital_human_audio: predecessor Clip %d is not ready: %s"
                        % (predecessor, reason)
                    )
                source_latent, predecessor_path = _load_indexed_av_checkpoint(
                    managed_prefix, predecessor
                )
                source_video, _source_audio = _streams_from_latent(source_latent)
                available = _pixel_frames(int(source_video.shape[2]))
                managed_context = _snap_context_length(
                    context_length, available, target_frames
                )
                clip_start_seconds = (
                    (managed_index - 1) * (target_frames - managed_context) / FPS
                )
                _LOG.info(
                    "h3_digital_human_audio: Clip %d loaded verified predecessor Clip %d from %s",
                    managed_index,
                    predecessor,
                    predecessor_path,
                )
        # The target latent is authoritative.  Native ComfyUI allocates the H3
        # audio stream by rounding the video duration onto the 40 Hz audio grid.
        # That grid can extend a fraction of one audio token past the final pixel
        # frame (e.g. 124f @ 24 fps = 206.666... ticks -> 207 target ticks).
        expected_audio_steps = int(target_audio.shape[-1])
        calculated_audio_steps = int(round(target_frames / FPS * AUDIO_HZ))
        if expected_audio_steps != calculated_audio_steps:
            _LOG.warning(
                "h3_digital_human_audio: target latent has %d audio steps for %d video frames; "
                "the nominal 40 Hz calculation gives %d. Using the target latent length.",
                expected_audio_steps,
                target_frames,
                calculated_audio_steps,
            )

        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        waveform = _stereo_first_batch(master_audio["waveform"], "master_audio")
        waveform = _resample_waveform(waveform, int(master_audio["sample_rate"]), vae_sr, "master_audio")
        clip_start_seconds = float(clip_start_seconds)
        start_sample = sample_boundary_from_seconds(clip_start_seconds, vae_sr)
        if start_sample < 0:
            raise ValueError("h3_digital_human_audio: clip_start_seconds must be >= 0")
        # Derive BOTH endpoints from the absolute master-audio timeline.
        # Do not round the clip duration separately and add it to start_sample:
        # at rates such as 32 kHz / 24 fps that can move an endpoint by one
        # sample on some clip indices.  Absolute endpoints guarantee that the
        # conditioning slice occupies exactly the same timeline interval as
        # this raw video's pixel frames.
        picture_end_seconds = clip_start_seconds + target_frames / FPS
        picture_end_sample = sample_boundary_from_seconds(picture_end_seconds, vae_sr)
        picture_samples = picture_end_sample - start_sample
        if picture_samples < 1:
            raise RuntimeError("h3_digital_human_audio: computed an empty master-audio slice")
        waveform_samples = int(waveform.shape[-1])
        if start_sample >= waveform_samples:
            raise ValueError(
                "h3_digital_human_audio: clip starts at %.3fs, at or after the master "
                "audio endpoint %.3fs; check the automatic timeline and managed clip index"
                % (clip_start_seconds, waveform_samples / float(vae_sr))
            )
        audio_slice = waveform[..., start_sample:picture_end_sample]
        audio_slice = _fit_waveform(audio_slice, picture_samples, "master audio slice")

        # Encode on the target H3 audio grid, not the picture-duration PCM
        # span.  This keeps clip_start_seconds exact and makes ComfyUI's generic
        # VAE center-crop a no-op.  The rounded 40-Hz target can end slightly
        # before or after the last pixel-frame boundary.
        _grid_sr, samples_per_latent, grid_samples = audio_grid_geometry(
            audio_vae, expected_audio_steps
        )
        if _grid_sr != vae_sr:
            raise RuntimeError(
                "h3_digital_human_audio: audio VAE grid reports %d Hz but audio_sample_rate is %d"
                % (_grid_sr, vae_sr)
            )
        encode_end_sample = start_sample + grid_samples
        encode_slice = waveform[..., start_sample:min(encode_end_sample, waveform_samples)]
        pcm_pad = grid_samples - int(encode_slice.shape[-1])
        if pcm_pad > 0:
            # The final long-video clip normally extends past the soundtrack:
            # for example five 10.125 s clips with 39-frame overlaps cover
            # 44.125 s for a 40 s master.  Silence-pad that final conditioning
            # window; the checkpoint assembler keeps the untouched master audio
            # and trims the completed video back to its exact duration.  A clip
            # whose *start* is already past the endpoint is rejected above, so
            # an invalid managed clip index cannot manufacture wholly silent
            # continuation clips.
            _LOG.info(
                "h3_digital_human_audio: final clip extends %.3fs beyond master audio; "
                "zero-padding conditioning tail (final assembly trims to master)",
                pcm_pad / float(vae_sr),
            )
            encode_slice = torch.nn.functional.pad(encode_slice, (0, pcm_pad))

        audio_full, _grid_diag = encode_exact_audio_grid(
            audio_vae, encode_slice, expected_audio_steps, "h3_digital_human_audio: master audio"
        )
        _LOG.debug(
            "h3_digital_human_audio: exact audio grid %d steps x %d samples; picture delta %+d samples; tail pad %d",
            expected_audio_steps,
            samples_per_latent,
            grid_samples - picture_samples,
            max(0, pcm_pad),
        )

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_audio.copy_(audio_full[:1].to(device=out_audio.device, dtype=out_audio.dtype))

        video_steps = 0
        n = 0
        if source_latent is not None and source_frames is not None:
            raise ValueError(
                "h3_digital_human_audio: connect either source_latent or source_frames, not both"
            )

        if source_latent is not None:
            source_video, _source_audio = _streams_from_latent(source_latent)
            if int(source_video.shape[0]) != 1:
                raise ValueError("h3_digital_human_audio: source_latent batch size 1 only")
            if int(context_length) <= 0:
                raise ValueError(
                    "h3_digital_human_audio: context_length must be > 0 when source_latent is connected"
                )
            if (
                int(source_video.shape[4]) != int(target_video.shape[4])
                or int(source_video.shape[3]) != int(target_video.shape[3])
                or int(source_video.shape[1]) != int(target_video.shape[1])
            ):
                raise ValueError(
                    "h3_digital_human_audio: source_latent and target latent video shapes/resolution must match"
                )

            available = _pixel_frames(int(source_video.shape[2]))
            n = _snap_context_length(context_length, available, target_frames)
            video_steps = _video_steps_for_frames(n)
            if video_steps >= int(target_video.shape[2]):
                raise ValueError("h3_digital_human_audio: video context consumes the whole target latent")
            if video_steps > int(source_video.shape[2]):
                raise RuntimeError("h3_digital_human_audio: source_latent is shorter than requested context")

            tail_start = int(source_video.shape[2]) - video_steps
            if tail_start % len(FRAME_PER_TOKEN) != 0:
                raise ValueError(
                    "h3_digital_human_audio: source_latent tail begins at H3 temporal phase %d; "
                    "use an H3-safe source length or the source_frames fallback"
                    % (tail_start % len(FRAME_PER_TOKEN))
                )
            vp = source_video[:, :, tail_start:, :, :].to(
                device=out_video.device, dtype=out_video.dtype
            )
            out_video[:, :, :video_steps] = vp

        elif source_frames is not None:
            if vae is None:
                raise ValueError("h3_digital_human_audio: vae is required when source_frames is connected")
            if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
                raise ValueError("h3_digital_human_audio: source_frames must be IMAGE [N,H,W,C]")
            if int(context_length) <= 0:
                raise ValueError("h3_digital_human_audio: context_length must be > 0 when source_frames is connected")
            idx = _cfr_index_map(int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS)
            available = int(idx.numel())
            n = _snap_context_length(context_length, available, target_frames)
            tail_idx = idx[-n:]
            video_tail = source_frames.index_select(0, tail_idx)
            video_tail = _resize_images(video_tail, width, height, crop)
            video_prefix = vae.encode(video_tail)
            if getattr(video_prefix, "ndim", 0) != 5:
                raise ValueError(
                    "h3_digital_human_audio: video VAE returned %s, expected [B,C,T,H,W]"
                    % (tuple(getattr(video_prefix, "shape", ())),)
                )
            video_steps = int(video_prefix.shape[2])
            covered = _pixel_frames(video_steps)
            if covered != n:
                raise RuntimeError(
                    "h3_digital_human_audio: %d context frames encoded to %d video latent steps covering %d frames; refusing a phase-shifted seam"
                    % (n, video_steps, covered)
                )
            if video_steps >= int(target_video.shape[2]):
                raise ValueError("h3_digital_human_audio: video context consumes the whole target latent")
            vp = video_prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
            if tuple(vp.shape[1:2] + vp.shape[3:]) != tuple(out_video.shape[1:2] + out_video.shape[3:]):
                raise ValueError(
                    "h3_digital_human_audio: encoded video prefix shape %s does not match target %s"
                    % (tuple(vp.shape), tuple(out_video.shape))
                )
            out_video[:, :, :video_steps] = vp

        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device,
            dtype=torch.float32,
        )
        if video_steps > 0:
            video_mask[:, :, :video_steps] = 0.0

        audio_mask = torch.zeros(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device,
            dtype=torch.float32,
        )

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        clip_audio = {"waveform": audio_slice, "sample_rate": vae_sr}
        _LOG.info(
            "h3_digital_human_audio: target %d frames (%.3fs); exact audio slice %.3fs -> %.3fs encoded to %d audio latent steps and fully preserved; visual prefix %d frames -> %d video steps",
            target_frames,
            target_frames / FPS,
            float(clip_start_seconds),
            float(clip_start_seconds) + target_frames / FPS,
            expected_audio_steps,
            n,
            video_steps,
        )
        return (out, n, clip_audio)
