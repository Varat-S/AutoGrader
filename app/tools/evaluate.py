from typing import Tuple, List, Union, Optional
import numpy as np
from app.models.analysis import ShotMetrics
from app.models.grade import ConsistencyScore
from app.media.ffmpeg import probe_video, extract_sampled_frames
from app.media.color import aggregate_shot_metrics, compute_consistency_score

def evaluate_grade(
    reference_metrics: ShotMetrics,
    graded_video_or_frames: Union[str, List[np.ndarray]],
    evaluation_mode: str = "same_scene_match",
    timestamps: Optional[List[float]] = None,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    duration_sec: float = 3.0
) -> Tuple[ShotMetrics, ConsistencyScore]:
    if isinstance(graded_video_or_frames, str):
        # Rendered video path
        info = probe_video(graded_video_or_frames)
        frames, ts = extract_sampled_frames(graded_video_or_frames)
        graded_metrics = aggregate_shot_metrics(
            shot_id="graded_output",
            video_path=graded_video_or_frames,
            frames=frames,
            timestamps=ts,
            fps=info.get("fps", 30.0),
            width=info.get("width", 1920),
            height=info.get("height", 1080),
            duration_sec=info.get("duration_sec", 3.0)
        )
    else:
        # Fast proxy / sampled frames in memory
        frames = graded_video_or_frames
        ts = timestamps if timestamps else [float(i) for i in range(len(frames))]
        graded_metrics = aggregate_shot_metrics(
            shot_id="graded_preview",
            video_path="",
            frames=frames,
            timestamps=ts,
            fps=fps,
            width=width,
            height=height,
            duration_sec=duration_sec
        )
        
    score = compute_consistency_score(
        reference=reference_metrics,
        candidate=graded_metrics,
        evaluation_mode=evaluation_mode
    )
    return graded_metrics, score