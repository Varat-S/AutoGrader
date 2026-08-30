from typing import Optional, List
import numpy as np
from app.models.analysis import ShotMetrics
from app.media.ffmpeg import probe_video, extract_sampled_frames
from app.media.color import aggregate_shot_metrics

def measure_shot_color(
    video_path: str,
    shot_id: str,
    fractions: List[float] = [0.25, 0.50, 0.75]
) -> ShotMetrics:
    info = probe_video(video_path)
    frames, timestamps = extract_sampled_frames(video_path, fractions=fractions)
    metrics = aggregate_shot_metrics(
        shot_id=shot_id,
        video_path=video_path,
        frames=frames,
        timestamps=timestamps,
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        duration_sec=info["duration_sec"]
    )
    return metrics