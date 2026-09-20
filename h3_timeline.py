"""Pure timeline arithmetic for the reusable H3 digital-human pipeline."""

from __future__ import annotations

import math


FPS = 24.0


def calculate_audio_timeline(
    sample_count,
    sample_rate,
    raw_clip_frames=362,
    context_frames=39,
    automatic=True,
    manual_clips=1,
    max_clips=20,
):
    samples = int(sample_count)
    rate = int(sample_rate)
    raw = int(raw_clip_frames)
    context = int(context_frames)
    limit = int(max_clips)
    if samples < 1 or rate < 1:
        raise ValueError("h3_timeline_plan: audio samples and sample rate must be positive")
    if raw < 5 or (raw - 5) % 17:
        raise ValueError(
            "h3_timeline_plan: raw_clip_frames must be an H3 run: 5, 22, 39, ..."
        )
    if context < 5 or (context - 5) % 17:
        raise ValueError(
            "h3_timeline_plan: context_frames must be an H3 run: 5, 22, 39, ..."
        )
    if context >= raw:
        raise ValueError("h3_timeline_plan: context_frames must be shorter than raw clip")
    if limit < 1:
        raise ValueError("h3_timeline_plan: max_clips must be positive")

    advance = raw - context
    required_frames = int(math.ceil(samples * FPS / rate))
    automatic_count = 1 + int(math.ceil(max(0, required_frames - raw) / advance))
    count = automatic_count if bool(automatic) else max(1, int(manual_clips))
    capacity_frames = raw + (limit - 1) * advance
    if count > limit:
        raise ValueError(
            "h3_timeline_plan: audio needs %d clips, but this workflow supports "
            "at most %d (about %.3fs with the current frame settings)"
            % (count, limit, capacity_frames / FPS)
        )
    coverage_frames = raw + (count - 1) * advance
    return {
        "clip_count": count,
        "automatic_clip_count": automatic_count,
        "duration_seconds": samples / float(rate),
        "coverage_seconds": coverage_frames / FPS,
        "raw_clip_frames": raw,
        "context_frames": context,
        "advance_frames": advance,
    }
