from typing import Tuple
from app.models.analysis import ShotMetrics
from app.models.grade import ConsistencyScore
from app.media.ffmpeg import probe_video, extract_sampled_frames
from app.media.color import aggregate_shot_metrics, compute_consistency_score

def evaluate_grade(
    reference_metrics: ShotMetrics,
    graded_video_path: str
) -> Tuple[ShotMetrics, ConsistencyScore]:
    info = probe_video(graded_video_path)
    frames, timestamps = extract_sampled_frames(graded_video_path)
    graded_metrics = aggregate_shot_metrics(
        shot_id="graded_output",
        video_path=graded_video_path,
        frames=frames,
        timestamps=timestamps,
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        duration_sec=info["duration_sec"]
    )
    score = compute_consistency_score(reference_metrics, graded_metrics)
    return graded_metrics, score