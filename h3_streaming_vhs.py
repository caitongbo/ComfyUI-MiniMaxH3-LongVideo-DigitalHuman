"""Single-execution low-RAM final H3 video output backed by Video Helper Suite.

The normal ComfyUI IMAGE datatype is a materialized ``torch.Tensor``.  Long H3
runs can therefore require tens of GiB just to hold the final RGB movie before
VHS sees it.  These output nodes keep the frame stream internal instead:

    H3 latent -> decode one clip -> resolve seam -> VHS encoder -> release clip

No custom object is exposed as a ComfyUI IMAGE output. The internal one-shot sequence exists only during the call into VHS_VideoCombine.
"""

from __future__ import annotations

import inspect
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import PurePosixPath
import logging

import torch

try:
    from safetensors import safe_open as _safe_open
except ImportError:  # ComfyUI normally provides safetensors.
    _safe_open = None

# Keep this module importable even in lightweight/mock test environments that
# replace the implementation modules with partial stubs.  Runtime helpers are
# resolved only when a streaming node actually executes.
FPS = 24.0

def _digital_human():
    from . import h3_digital_human_audio_context as module
    return module

def _checkpoint_manifest():
    from . import checkpoint_manifest as module
    return module

def _media():
    from . import h3_media_utils as module
    return module

def _decode_h3_video_cpu(*args, **kwargs):
    return _media().decode_h3_video_cpu(*args, **kwargs)

def _pixel_frames(latent_steps):
    return _media().pixel_frames(latent_steps)

def _release_decode_memory():
    return _media().release_decode_memory()

def _snap_digital_human_context_length(*args, **kwargs):
    return _digital_human()._snap_context_length(*args, **kwargs)


_LOG = logging.getLogger("h3_digital_human.streaming_vhs")


def _rgb_gib(frames, height, width):
    return int(frames) * int(height) * int(width) * 3 * 4 / float(1024 ** 3)


def _available_memory_gib():
    """Best-effort currently available host memory, not total installed RAM."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / float(1024 ** 3)
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * size / float(1024 ** 3)
    except (AttributeError, OSError, ValueError):
        return None


def _temporary_directory():
    try:
        import folder_paths

        path = folder_paths.get_temp_directory()
    except Exception:
        path = tempfile.gettempdir()
    os.makedirs(path, exist_ok=True)
    return path


def _new_decode_buffer_path():
    descriptor, path = tempfile.mkstemp(
        prefix=".h3_checkpoint_decode_",
        suffix=".buffer",
        dir=_temporary_directory(),
    )
    os.close(descriptor)
    return path


def _enforce_decode_disk_budget(shapes, decode_buffer):
    """Ensure a disk-backed RGB buffer cannot fill the temp filesystem."""
    if str(decode_buffer).strip().lower() != "disk-backed":
        return None, None
    height = int(shapes[0][3]) * 16
    width = int(shapes[0][4]) * 16
    largest_frames = max(_pixel_frames(shape[2]) for shape in shapes)
    required = _rgb_gib(largest_frames, height, width)
    temp_dir = _temporary_directory()
    free = shutil.disk_usage(temp_dir).free / float(1024 ** 3)
    # Leave room for filesystem metadata, the encoder output and unrelated
    # ComfyUI temp work. The mmap itself is deleted after every clip.
    reserve = max(2.0, required * 0.25)
    if required + reserve > free:
        raise OSError(
            "h3_checkpoint_digital_human: disk-backed VAE decode needs about %.2f GiB "
            "for its largest clip plus %.2f GiB reserve, but the ComfyUI temp "
            "filesystem has only %.2f GiB free (%s). Free disk space, shorten "
            "the raw clip, lower resolution, or select the guarded memory "
            "decode mode."
            % (required, reserve, free, temp_dir)
        )
    return required, free


def _output_checkpoint_paths(checkpoint_prefix, clip_count):
    """Resolve fixed indexed Save Latent slots inside ComfyUI output only."""
    try:
        import folder_paths
    except Exception as exc:  # pragma: no cover - broken ComfyUI install
        raise RuntimeError(
            "h3_checkpoint_digital_human: ComfyUI folder_paths is unavailable"
        ) from exc

    prefix = str(checkpoint_prefix).strip().replace("\\", "/")
    pure = PurePosixPath(prefix)
    if not prefix or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(
            "h3_checkpoint_digital_human: checkpoint_prefix must be a relative prefix "
            "inside the ComfyUI output directory"
        )
    if prefix.endswith(".safetensors"):
        prefix = prefix[:-len(".safetensors")]

    output = os.path.realpath(folder_paths.get_output_directory())
    base = os.path.realpath(os.path.join(output, *PurePosixPath(prefix).parts))
    if os.path.commonpath((output, base)) != output:
        raise ValueError("h3_checkpoint_digital_human: checkpoint prefix escapes output")

    paths = [f"{base}_{index:05d}.safetensors"
             for index in range(1, int(clip_count) + 1)]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        index = paths.index(missing[0]) + 1
        raise FileNotFoundError(
            "h3_checkpoint_digital_human: Clip %d checkpoint is missing: %s. "
            "Use Run/resume checkpoints with the same project prefix and clip count."
            % (index, missing[0])
        )
    return paths


def _checkpoint_video_shape(path):
    if _safe_open is None:
        raise RuntimeError(
            "h3_checkpoint_digital_human: safetensors is unavailable; checkpoints "
            "cannot be inspected"
        )
    try:
        with _safe_open(path, framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            if not {"video", "audio"}.issubset(keys):
                raise ValueError("missing video/audio tensors")
            shape = tuple(int(value) for value in tensors.get_slice("video").get_shape())
    except Exception as exc:
        raise ValueError(
            "h3_checkpoint_digital_human: invalid or incomplete checkpoint %s: %s"
            % (path, exc)
        ) from exc
    if len(shape) != 5 or shape[0] != 1:
        raise ValueError(
            "h3_checkpoint_digital_human: %s video tensor must be [1,C,T,H,W], got %s"
            % (path, shape)
        )
    return shape


def _checkpoint_digital_human_plan(paths, context_frames):
    shapes = [_checkpoint_video_shape(path) for path in paths]
    first = shapes[0]
    for index, shape in enumerate(shapes[1:], start=2):
        if (shape[1], shape[3], shape[4]) != (first[1], first[3], first[4]):
            raise ValueError(
                "h3_checkpoint_digital_human: Clip %d latent geometry %s does not "
                "match Clip 1 %s" % (index, shape, first)
            )
    raw_frames = [_pixel_frames(shape[2]) for shape in shapes]
    contexts = [0]
    for index in range(1, len(raw_frames)):
        contexts.append(_snap_digital_human_context_length(
            int(context_frames), raw_frames[index - 1], raw_frames[index]
        ))
    return shapes, raw_frames, contexts


def _disk_backed_decode_available(video_vae, video_channels):
    first_stage = getattr(video_vae, "first_stage_model", None)
    latent_mean = getattr(first_stage, "latents_mean", None)
    return bool(
        first_stage is not None
        and bool(getattr(first_stage, "comfy_has_chunked_io", False))
        and callable(getattr(first_stage, "decode_output_shape", None))
        and callable(getattr(first_stage, "decode", None))
        and torch.is_tensor(latent_mean)
        and int(latent_mean.numel()) == int(video_channels)
        and callable(getattr(video_vae, "memory_used_decode", None))
        and callable(getattr(video_vae, "throw_exception_if_invalid", None))
        and hasattr(video_vae, "patcher")
        and hasattr(video_vae, "device")
        and hasattr(video_vae, "vae_dtype")
        and hasattr(video_vae, "disable_offload")
    )


def _enforce_decode_memory_budget(
    shapes, raw_frames, contexts, overlap, requested_gib,
    disk_backed_active=False,
):
    """Refuse a likely host OOM before VAE decode can trigger the OS killer."""
    height = int(shapes[0][3]) * 16
    width = int(shapes[0][4]) * 16
    seam_frames = max(_seam_overlaps(raw_frames, contexts, overlap))
    largest = max(_rgb_gib(frames, height, width) for frames in raw_frames)
    seam = _rgb_gib(seam_frames, height, width)
    # Full decoded RGB, conversion/copy headroom, the retained seam, and a
    # fixed allowance for encoder queues/Python. This is intentionally
    # conservative: a clear preflight error is preferable to SIGKILL.
    estimated = (
        max(1.5, seam * 2.0 + 1.0)
        if disk_backed_active
        else largest * 1.75 + seam + 0.75
    )
    available = _available_memory_gib()
    requested = float(requested_gib)
    if requested > 0:
        budget = requested
        if available is not None:
            budget = min(budget, available * 0.80)
    elif available is not None:
        budget = available * 0.70
    else:
        budget = 8.0
    if estimated > budget:
        raise MemoryError(
            "h3_checkpoint_digital_human: refusing final decode before the OS can kill "
            "ComfyUI. Estimated peak host RAM %.2f GiB exceeds the %.2f GiB "
            "assembly budget (available now: %s). Use shorter raw clips, a "
            "lower resolution, free RAM, or explicitly choose a larger budget "
            "only when the machine really has it."
            % (estimated, budget,
               "unknown" if available is None else "%.2f GiB" % available)
        )
    return estimated, budget, available


def _decode_checkpoint_video(
    video_vae, video_latent, decode_buffer, buffer_path
):
    """Decode one checkpoint, preferring a file-backed H3 output buffer."""
    mode = str(decode_buffer or "disk-backed").strip().lower()
    if mode not in ("disk-backed", "memory"):
        raise ValueError(
            "h3_checkpoint_digital_human: decode_buffer must be disk-backed or memory"
        )
    if not torch.is_tensor(video_latent) or video_latent.ndim != 5:
        raise ValueError(
            "h3_checkpoint_digital_human: checkpoint video latent must be [B,C,T,H,W]"
        )

    if mode == "disk-backed" and _disk_backed_decode_available(
        video_vae, int(video_latent.shape[1])
    ):
        from comfy import model_management

        first_stage = video_vae.first_stage_model
        output_shape = tuple(
            int(part)
            for part in first_stage.decode_output_shape(video_latent.shape)
        )
        if len(output_shape) != 5 or output_shape[0] != 1:
            raise ValueError(
                "h3_checkpoint_digital_human: disk-backed VAE output must be "
                "[1,C,T,H,W], got %s" % (output_shape,)
            )
        mapped = torch.from_file(
            buffer_path,
            shared=True,
            size=math.prod(output_shape),
            dtype=torch.float32,
        ).reshape(output_shape)
        video_vae.throw_exception_if_invalid()
        memory_used = video_vae.memory_used_decode(
            video_latent.shape, video_vae.vae_dtype
        )
        model_management.load_models_gpu(
            [video_vae.patcher],
            memory_required=memory_used,
            force_full_load=video_vae.disable_offload,
        )
        with torch.inference_mode(), model_management.cuda_device_context(
            video_vae.device
        ):
            samples = video_latent.to(
                device=video_vae.device, dtype=video_vae.vae_dtype
            )
            first_stage.decode(samples, output_buffer=mapped)
            del samples
        images = mapped.movedim(1, -1)[0]
        return images, mapped, "disk-backed mmap"

    if mode == "disk-backed":
        _LOG.warning(
            "h3_checkpoint_digital_human: this H3 VAE/ComfyUI build does not expose "
            "chunked output_buffer decode; falling back to guarded one-clip RAM"
        )
    with torch.inference_mode():
        images = _decode_h3_video_cpu(video_vae, video_latent)
    return images, None, "one-clip memory"


def _resolve_vhs_video_combine():
    """Resolve VHS at execution time without making plugin import order fragile."""
    try:
        import nodes as comfy_nodes
    except Exception as exc:  # pragma: no cover - only reachable in broken Comfy installs
        raise RuntimeError(
            "h3_streaming_vhs: could not import ComfyUI's nodes module"
        ) from exc

    cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("VHS_VideoCombine")
    if cls is None:
        raise RuntimeError(
            "h3_streaming_vhs: VHS_VideoCombine is required. Install/enable "
            "ComfyUI-VideoHelperSuite before using the streaming final output."
        )

    combine = getattr(cls, "combine_video", None)
    if not callable(combine):
        raise RuntimeError(
            "h3_streaming_vhs: the installed VHS_VideoCombine node does not expose "
            "the expected combine_video API. Update ComfyUI-VideoHelperSuite."
        )

    # Fail with a useful message instead of a long TypeError if an older VHS
    # build is missing controls used by the direct-stream nodes. Future/wrapped
    # implementations that accept **kwargs remain compatible.
    try:
        parameters = inspect.signature(combine).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        expected = {
            "frame_rate", "loop_count", "images", "filename_prefix", "format",
            "pingpong", "save_output", "prompt", "extra_pnginfo", "audio",
            "unique_id", "pix_fmt", "crf", "save_metadata", "trim_to_audio",
        }
        missing = sorted(expected.difference(parameters))
        if missing:
            raise RuntimeError(
                "h3_streaming_vhs: the installed VideoHelperSuite is too old for "
                "direct H3 streaming; VHS_VideoCombine.combine_video is missing: "
                + ", ".join(missing)
                + ". Update ComfyUI-VideoHelperSuite."
            )
    return cls


class _OneShotFrameSequence:
    """Sequence facade for VHS that primes once and then streams exactly once.

    VHS currently asks for ``len(images)``, reads ``images[0]`` for output
    dimensions/metadata, and then iterates the image source.  Priming the
    generator in ``__getitem__(0)`` avoids decoding Clip 1 twice.
    """

    def __init__(self, frame_count, generator_factory):
        self._frame_count = int(frame_count)
        self._generator_factory = generator_factory
        self._generator = None
        self._first = None
        self._primed = False
        self._iterated = False

    def __len__(self):
        return self._frame_count

    def _prime(self):
        if self._primed:
            return
        self._generator = iter(self._generator_factory())
        try:
            self._first = next(self._generator).detach().clone()
        except StopIteration as exc:
            raise RuntimeError("h3_streaming_vhs: frame stream produced no frames") from exc
        self._primed = True

    def __getitem__(self, index):
        if int(index) != 0:
            raise IndexError(
                "h3_streaming_vhs: internal frame stream only supports the first-frame probe"
            )
        self._prime()
        return self._first

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("h3_streaming_vhs: frame stream is one-shot")
        self._prime()
        self._iterated = True
        first = self._first
        self._first = None
        yield first
        yield from self._generator


def _yield_segments_and_hold(segments, hold_frames, detach_frames=False):
    """Yield all but the final ``hold_frames`` and return a detached CPU tail.

    The returned tail is the only RGB history kept for the next seam.  It is
    cloned so it does not keep the much larger decoded clip storage alive.
    """
    segments = [seg for seg in segments if seg is not None and int(seg.shape[0]) > 0]
    total = sum(int(seg.shape[0]) for seg in segments)
    hold = max(0, min(int(hold_frames), total))
    emit = total - hold
    remainders = []

    for seg in segments:
        n = int(seg.shape[0])
        take = min(n, emit)
        if take > 0:
            for frame in seg[:take]:
                # Disk-backed decodes must not hand mmap views to VHS: its
                # loop may retain the last item while requesting the next one,
                # preventing prompt buffer deletion on Windows. One detached
                # frame keeps memory bounded without retaining the scene map.
                yield frame.detach().clone() if detach_frames else frame
            emit -= take
        if take < n:
            remainders.append(seg[take:])

    if emit != 0:
        raise RuntimeError("h3_streaming_vhs: internal frame accounting error")
    if hold == 0:
        return None

    if not remainders:
        raise RuntimeError("h3_streaming_vhs: failed to retain requested seam tail")
    if len(remainders) == 1:
        tail = remainders[0].detach().to(device="cpu", dtype=torch.float32).clone()
    else:
        tail = torch.cat(remainders, dim=0).detach().to(device="cpu", dtype=torch.float32)
    if int(tail.shape[0]) != hold:
        raise RuntimeError(
            f"h3_streaming_vhs: retained {int(tail.shape[0])} frames, expected {hold}"
        )
    return tail.contiguous()


def _seam_overlaps(raw_frames, contexts, overlap):
    """Return overlap frames for each seam using the existing assembler rules."""
    overlaps = [0] * len(raw_frames)
    write_frame = int(raw_frames[0])
    overlap = max(0, int(overlap))
    for i in range(1, len(raw_frames)):
        ov = min(overlap, int(contexts[i]), write_frame)
        overlaps[i] = ov
        write_frame += int(raw_frames[i]) - int(contexts[i])
    return overlaps
def _checkpoint_frame_generator(
    video_vae, paths, raw_frames, contexts, overlap, decode_buffer, log_prefix
):
    """Load, decode, emit and release exactly one checkpoint at a time."""
    seam_ovs = _seam_overlaps(raw_frames, contexts, overlap)
    tail = None

    for i, path in enumerate(paths):
        buffer_path = _new_decode_buffer_path()
        video_latent = decoded = mapped = segments = new_tail = None
        try:
            # Read only the video tensor. The checkpoint audio remains on disk;
            # final Digital Human Video output muxes the untouched master waveform.
            with _safe_open(path, framework="pt", device="cpu") as tensors:
                video_latent = tensors.get_tensor("video")
            decoded, mapped, decoder_mode = _decode_checkpoint_video(
                video_vae, video_latent, decode_buffer, buffer_path
            )
            _LOG.info(
                "%s: Clip %d/%d decode mode: %s",
                log_prefix, i + 1, len(paths), decoder_mode,
            )
            if int(decoded.shape[0]) != int(raw_frames[i]):
                raise RuntimeError(
                    f"{log_prefix}: Clip {i + 1} video decode produced "
                    f"{int(decoded.shape[0])} frames; expected {int(raw_frames[i])}"
                )

            if i == 0:
                segments = [decoded]
            else:
                ctx = int(contexts[i])
                ov = int(seam_ovs[i])
                if ov > 0:
                    if tail is None or int(tail.shape[0]) != ov:
                        raise RuntimeError(
                            f"{log_prefix}: seam {i} retained tail mismatch "
                            f"({0 if tail is None else int(tail.shape[0])} != {ov})"
                        )
                    dst = decoded[ctx - ov:ctx]
                    alpha = torch.linspace(
                        0.0, 1.0, ov + 2, dtype=torch.float32, device="cpu"
                    )[1:-1].view(-1, 1, 1, 1)
                    tail.mul_(1.0 - alpha).add_(dst * alpha)
                    del dst, alpha
                    segments = [tail, decoded[ctx:]]
                else:
                    segments = [decoded[ctx:]]
                    tail = None

            next_hold = int(seam_ovs[i + 1]) if i + 1 < len(paths) else 0
            new_tail = yield from _yield_segments_and_hold(
                segments, next_hold, detach_frames=(mapped is not None)
            )
            tail = new_tail
        except Exception as exc:
            if isinstance(exc, (MemoryError, RuntimeError, ValueError)):
                raise
            raise RuntimeError(
                f"{log_prefix}: failed to process Clip {i + 1} checkpoint "
                f"{path}: {exc}"
            ) from exc
        finally:
            # The mmap must be released before unlink on Windows. Explicitly
            # clear every view even when VHS/FFmpeg aborts while consuming it.
            decoded = segments = new_tail = video_latent = mapped = None
            _release_decode_memory()
            try:
                os.unlink(buffer_path)
            except FileNotFoundError:
                pass

    if tail is not None:
        raise RuntimeError(f"{log_prefix}: unflushed seam tail after final clip")
def _vhs_h264_inputs(filename_default, trim_default):
    return {
        "filename_prefix": ("STRING", {"default": filename_default}),
        "pix_fmt": (["yuv420p", "yuv420p10le", "lossless_ffv1"], {"default": "yuv420p"}),
        "crf": ("INT", {"default": 19, "min": 0, "max": 100, "step": 1}),
        "save_metadata": ("BOOLEAN", {"default": False}),
        "trim_to_audio": ("BOOLEAN", {"default": bool(trim_default)}),
        "save_output": ("BOOLEAN", {"default": True}),
    }


def _expand_output_prefix(prefix, now=None):
    """Resolve Comfy-style date folders for UI and API callers (server local time)."""
    now = now or datetime.now()
    values = {"yyyy": f"{now.year:04d}", "yy": f"{now.year % 100:02d}",
              "MM": f"{now.month:02d}", "dd": f"{now.day:02d}",
              "HH": f"{now.hour:02d}", "hh": f"{now.hour % 12 or 12:02d}",
              "mm": f"{now.minute:02d}", "ss": f"{now.second:02d}"}
    def expand(match):
        pattern = match.group(1)
        # Only documented date tokens and safe separators are supported.
        parts = re.findall(r"yyyy|yy|MM|dd|HH|hh|mm|ss|[^A-Za-z]", pattern)
        if "".join(parts) != pattern:
            raise ValueError("Unsupported date token; use yyyy, yy, MM, dd, HH, hh, mm, ss.")
        return "".join(values.get(part, part) for part in parts)
    value = re.sub(r"%date:([^%]+)%", expand, str(prefix)).replace("\\", "/")
    if (not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or re.search(r"[<>:\x00-\x1f]", value) or "%date:" in value):
        raise ValueError("Output prefix must be a relative filename/subfolder inside ComfyUI output.")
    return value


def _run_vhs_h264(
    frames,
    audio,
    filename_prefix,
    pix_fmt,
    crf,
    save_metadata,
    trim_to_audio,
    save_output,
    prompt,
    extra_pnginfo,
    unique_id,
):
    vhs_cls = _resolve_vhs_video_combine()
    vhs = vhs_cls()

    # IMPORTANT: final filename allocation belongs to VHS.  Do not construct a
    # fixed output path in this module: VHS scans the destination directory and
    # advances its numeric counter for repeated runs using the same prefix.
    # Keeping that responsibility here would risk making repeated runs
    # overwrite the previous final video.
    # Keep the existing H.264 path as the default. ``lossless_ffv1`` is a
    # sentinel in the pix_fmt widget so its stable ordering and serialized
    # defaults remain unchanged. VHS's built-in FFV1 MKV format
    # supplies its own codec defaults; rgb48le keeps our 3-channel H3 frames in
    # 16-bit RGB without introducing an unnecessary alpha plane. FLAC is used
    # for audio by that format.
    lossless = str(pix_fmt) == "lossless_ffv1"
    output_format = "video/ffv1-mkv" if lossless else "video/h264-mp4"
    output_pix_fmt = "rgb48le" if lossless else str(pix_fmt)

    result = vhs.combine_video(
        frame_rate=24,
        loop_count=0,
        images=frames,
        filename_prefix=_expand_output_prefix(filename_prefix),
        format=output_format,
        pingpong=False,
        save_output=bool(save_output),
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
        audio=audio,
        unique_id=unique_id,
        pix_fmt=output_pix_fmt,
        crf=int(crf),
        save_metadata=bool(save_metadata),
        trim_to_audio=bool(trim_to_audio),
    )

    # VHS's own browser-side preview is only installed for nodes whose class
    # name is literally VHS_VideoCombine.  These H3 streaming nodes call VHS
    # internally, so preserve the normal VHS ``gifs`` UI payload and also expose
    # the same saved MP4 through ComfyUI's native PreviewVideo representation
    # (``images`` + ``animated``).  This restores a visible final video preview
    # without materializing the final RGB movie as an IMAGE tensor.
    if isinstance(result, dict):
        ui_payload = result.setdefault("ui", {})
        previews = ui_payload.get("gifs") or []
        if previews:
            preview = previews[0]
            if all(key in preview for key in ("filename", "type")):
                ui_payload["images"] = [{
                    "filename": preview["filename"],
                    "subfolder": preview.get("subfolder", ""),
                    "type": preview["type"],
                }]
                ui_payload["animated"] = (True,)
    return result
class MiniMaxH3StreamCheckpointDigitalHumanToVHS:
    """Assemble a Digital Human Video from atomic on-disk clip checkpoints.

    This node is intentionally disconnected from every sampler. Queue it as a
    separate top-level job after all indexed Save Latent nodes have completed,
    so ComfyUI does not retain the generation graph's live clip tensors while
    the video VAE decodes the final movie.
    """

    MAX_CLIPS = 64

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video_vae": ("VAE",),
            "master_audio": ("AUDIO",),
            "checkpoint_prefix": (
                "STRING",
                {
                    "default": "h3_digital_human/checkpoints/clip",
                    "tooltip": (
                        "Relative to ComfyUI output. Must match H3 Motion "
                        "Context Save Latent; indexed files are resolved as "
                        "<prefix>_00001.safetensors, _00002, and so on."
                    ),
                },
            ),
            "clip_count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": cls.MAX_CLIPS,
                    "tooltip": "Number of contiguous completed checkpoint slots to assemble.",
                },
            ),
            "context_frames": ("INT", {"default": 39, "min": 5, "max": 9999}),
            "video_overlap_frames": (
                "INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": (
                        "Use 0 for exact latent continuation: the preceding clip "
                        "owns the seam and the incoming protected prefix is discarded. "
                        "A positive value crossfades duplicate decoded frames and may "
                        "soften detail when the two decodes differ."
                    ),
                }
            ),
            "decode_buffer": (
                ["disk-backed", "memory"],
                {
                    "default": "disk-backed",
                    "tooltip": (
                        "disk-backed uses the H3 VAE chunked output_buffer API "
                        "to write decoded RGB into a temporary mmap instead of "
                        "the Python heap. Unsupported builds safely fall back "
                        "to guarded one-clip memory decode."
                    ),
                },
            ),
            "max_decode_ram_gb": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1024.0,
                    "step": 0.5,
                    "tooltip": (
                        "0 = automatically use at most 70% of currently "
                        "available host RAM. The node refuses an unsafe full-clip "
                        "VAE decode before the operating system can kill ComfyUI."
                    ),
                },
            ),
            "generation_retries": (
                "INT",
                {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "tooltip": (
                        "Frontend Run/resume manager retries a failed clip this "
                        "many times. The Python assembler does not retry FFmpeg."
                    ),
                },
            ),
            "cleanup_delay_ms": (
                "INT",
                {
                    "default": 1500,
                    "min": 0,
                    "max": 60000,
                    "step": 250,
                    "tooltip": (
                        "Delay after each successful top-level clip prompt so "
                        "ComfyUI can release model/allocator caches before the next."
                    ),
                },
            ),
        }
        required.update(_vhs_h264_inputs("video/minimax_h3_digital_human", True))
        return {
            "required": required,
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "stream_checkpoints_to_vhs"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Digital Human"
    DESCRIPTION = (
        "Kill-resistant final Digital Human Video assembly from atomic disk checkpoints. "
        "Queue as a separate job after generation; it loads only one clip video "
        "latent at a time, ignores checkpoint audio, muxes the untouched master "
        "audio, and refuses a predicted host-RAM OOM before decoding starts."
    )

    @classmethod
    def IS_CHANGED(cls, checkpoint_prefix, clip_count, **kwargs):
        try:
            paths = _output_checkpoint_paths(checkpoint_prefix, clip_count)
            return "|".join(
                "%s:%d:%d" % (path, os.stat(path).st_mtime_ns, os.stat(path).st_size)
                for path in paths
            )
        except Exception:
            return float("NaN")

    def stream_checkpoints_to_vhs(
        self,
        video_vae,
        master_audio,
        checkpoint_prefix="h3_digital_human/checkpoints/clip",
        clip_count=1,
        context_frames=39,
        video_overlap_frames=39,
        decode_buffer="disk-backed",
        max_decode_ram_gb=0.0,
        generation_retries=1,
        cleanup_delay_ms=1500,
        filename_prefix="video/minimax_h3_digital_human",
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=False,
        trim_to_audio=True,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        paths = _output_checkpoint_paths(checkpoint_prefix, clip_count)
        _checkpoint_manifest().validate_checkpoint_set(checkpoint_prefix, paths)
        shapes, raw_frames, contexts = _checkpoint_digital_human_plan(paths, context_frames)
        disk_backed_active = (
            str(decode_buffer).strip().lower() == "disk-backed"
            and _disk_backed_decode_available(video_vae, shapes[0][1])
        )
        disk_required, disk_free = _enforce_decode_disk_budget(
            shapes, decode_buffer if disk_backed_active else "memory"
        )
        estimated, budget, available = _enforce_decode_memory_budget(
            shapes,
            raw_frames,
            contexts,
            video_overlap_frames,
            max_decode_ram_gb,
            disk_backed_active=disk_backed_active,
        )
        final_frames = raw_frames[0] + sum(
            raw_frames[index] - contexts[index]
            for index in range(1, len(raw_frames))
        )

        waveform = master_audio["waveform"]
        if getattr(waveform, "ndim", 0) != 3:
            raise ValueError(
                "h3_checkpoint_digital_human: master_audio waveform must be [B,C,L], got %s"
                % (tuple(getattr(waveform, "shape", ())),)
            )
        audio = {
            "waveform": waveform[:1].detach().to(device="cpu").contiguous(),
            "sample_rate": int(master_audio["sample_rate"]),
        }

        factory = lambda: _checkpoint_frame_generator(
            video_vae,
            paths,
            raw_frames,
            contexts,
            video_overlap_frames,
            decode_buffer,
            "h3_checkpoint_digital_human",
        )
        frames = _OneShotFrameSequence(final_frames, factory)
        _LOG.info(
            "h3_checkpoint_digital_human: assembling %d atomic checkpoints / %d frames "
            "as a disk-backed final job; decoder=%s, estimated host-RAM peak "
            "%.2f GiB, budget %.2f GiB, available %s, mmap=%s",
            len(paths),
            final_frames,
            "disk-backed mmap" if disk_backed_active else "one-clip memory",
            estimated,
            budget,
            "unknown" if available is None else "%.2f GiB" % available,
            ("off" if disk_required is None else "%.2f/%.2f GiB" % (
                disk_required, disk_free)),
        )
        return _run_vhs_h264(
            frames,
            audio,
            filename_prefix,
            pix_fmt,
            crf,
            save_metadata,
            trim_to_audio,
            save_output,
            prompt,
            extra_pnginfo,
            unique_id,
        )
