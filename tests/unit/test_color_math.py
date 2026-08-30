import pytest
import numpy as np
from app.media.color import (
    compute_frame_metrics,
    aggregate_shot_metrics,
    calculate_deterministic_match_params,
    apply_color_grade_to_frame,
    compute_consistency_score
)
from app.models.grade import ColorGradeParams

def test_frame_metrics_computation():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    metrics = compute_frame_metrics(frame, timestamp_sec=1.0)
    
    assert abs(metrics.mean_luminance - 128.0) < 1.0
    assert abs(metrics.median_luminance - 128.0) < 1.0
    assert metrics.shadow_clip_pct == 0.0
    assert metrics.highlight_clip_pct == 0.0
    assert abs(metrics.lab_a_mean) < 1.0
    assert abs(metrics.lab_b_mean) < 1.0

def test_deterministic_exposure_match():
    ref_frame = np.full((100, 100, 3), 160, dtype=np.uint8)
    tgt_frame = np.full((100, 100, 3), 60, dtype=np.uint8)
    
    ref_shot = aggregate_shot_metrics('ref', 'ref.mp4', [ref_frame], [0.0], 30.0, 100, 100, 1.0)
    tgt_shot = aggregate_shot_metrics('tgt', 'tgt.mp4', [tgt_frame], [0.0], 30.0, 100, 100, 1.0)
    
    params = calculate_deterministic_match_params(ref_shot, tgt_shot)
    
    # Target is darker than reference -> lab_l_offset should be positive
    assert params.lab_l_offset > 0.0
    
    # Test grading application
    graded_frame = apply_color_grade_to_frame(tgt_frame, params)
    graded_metrics = compute_frame_metrics(graded_frame)
    
    # Graded luminance should be much closer to reference (160) than target (60)
    assert abs(graded_metrics.mean_luminance - 160.0) < abs(60.0 - 160.0)

def test_consistency_score():
    frame_a = np.full((100, 100, 3), 140, dtype=np.uint8)
    frame_b = np.full((100, 100, 3), 140, dtype=np.uint8)
    frame_c = np.full((100, 100, 3), 40, dtype=np.uint8)
    
    shot_a = aggregate_shot_metrics('a', 'a.mp4', [frame_a], [0.0], 30.0, 100, 100, 1.0)
    shot_b = aggregate_shot_metrics('b', 'b.mp4', [frame_b], [0.0], 30.0, 100, 100, 1.0)
    shot_c = aggregate_shot_metrics('c', 'c.mp4', [frame_c], [0.0], 30.0, 100, 100, 1.0)
    
    score_identical = compute_consistency_score(shot_a, shot_b)
    score_mismatched = compute_consistency_score(shot_a, shot_c)
    
    assert score_identical.overall_score > 95.0
    assert score_mismatched.overall_score < score_identical.overall_score
