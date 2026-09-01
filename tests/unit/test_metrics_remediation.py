import pytest
import numpy as np
from app.media.color import (
    compute_frame_metrics,
    aggregate_shot_metrics,
    compute_consistency_score,
    apply_color_grade_to_frame
)
from app.models.grade import GradePlan, ColorGradeParams

def test_identical_images_score_high():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    shot_ref = aggregate_shot_metrics("ref", "ref.mp4", [frame], [0.0], 30.0, 100, 100, 1.0)
    shot_cand = aggregate_shot_metrics("cand", "cand.mp4", [frame], [0.0], 30.0, 100, 100, 1.0)
    
    score = compute_consistency_score(shot_ref, shot_cand, evaluation_mode="same_scene_match")
    assert score.overall_score >= 98.0, f"Identical images should score >= 98, got {score.overall_score}"
    assert score.tonal_similarity >= 98.0
    assert score.chromatic_similarity >= 98.0
    assert score.clipping_health == 100.0

def test_dark_mismatch_cannot_outscore_identical():
    # Bright reference
    bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    # Dark candidate
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    
    shot_ref = aggregate_shot_metrics("ref", "ref.mp4", [bright_frame], [0.0], 30.0, 100, 100, 1.0)
    shot_identical = aggregate_shot_metrics("ident", "ident.mp4", [bright_frame], [0.0], 30.0, 100, 100, 1.0)
    shot_dark = aggregate_shot_metrics("dark", "dark.mp4", [dark_frame], [0.0], 30.0, 100, 100, 1.0)
    
    score_identical = compute_consistency_score(shot_ref, shot_identical, evaluation_mode="same_scene_match")
    score_dark = compute_consistency_score(shot_ref, shot_dark, evaluation_mode="same_scene_match")
    
    assert score_dark.tonal_similarity < 20.0, f"Dark mismatch should have very low tonal similarity, got {score_dark.tonal_similarity}"
    assert score_identical.overall_score > score_dark.overall_score + 50.0

def test_same_scene_underexposed_match():
    # Normal reference
    ref_frame = np.full((100, 100, 3), 140, dtype=np.uint8)
    # Underexposed candidate (~1.5 stops down)
    under_frame = np.full((100, 100, 3), 50, dtype=np.uint8)
    
    shot_ref = aggregate_shot_metrics("ref", "ref.mp4", [ref_frame], [0.0], 30.0, 100, 100, 1.0)
    shot_under = aggregate_shot_metrics("under", "under.mp4", [under_frame], [0.0], 30.0, 100, 100, 1.0)
    
    before_score = compute_consistency_score(shot_ref, shot_under, evaluation_mode="same_scene_match")
    assert before_score.tonal_similarity < 40.0
    
    # Apply balanced technical exposure correction (+1.48 EV)
    plan = GradePlan(shot_id="under")
    plan.technical_balance.exposure_ev = 1.48
    
    graded_frame = apply_color_grade_to_frame(under_frame, plan)
    shot_graded = aggregate_shot_metrics("graded", "graded.mp4", [graded_frame], [0.0], 30.0, 100, 100, 1.0)
    
    after_score = compute_consistency_score(shot_ref, shot_graded, evaluation_mode="same_scene_match")
    assert after_score.tonal_similarity > before_score.tonal_similarity + 40.0
    assert after_score.overall_score > before_score.overall_score

def test_day_night_cross_scene_continuity_independence():
    # Day reference: bright daylight
    day_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    day_frame[:, :] = [160, 150, 140] # Warm day
    
    # Night candidate: naturally dark scene with dark sky and practical light
    night_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    night_frame[:80, :] = [25, 20, 15] # Deep dark shadows
    night_frame[80:, :] = [90, 80, 70] # Practical light
    
    shot_day = aggregate_shot_metrics("day", "day.mp4", [day_frame], [0.0], 30.0, 100, 100, 1.0)
    shot_night = aggregate_shot_metrics("night", "night.mp4", [night_frame], [0.0], 30.0, 100, 100, 1.0)
    
    # In same_scene_match mode, night shot fails tonal match because it is naturally dark
    same_scene_score = compute_consistency_score(shot_day, shot_night, evaluation_mode="same_scene_match")
    assert same_scene_score.tonal_similarity < 20.0
    
    # In cross_scene_look_continuity mode, night shot is evaluated on look health & palette adherence, not absolute luminance
    cross_scene_score = compute_consistency_score(shot_day, shot_night, evaluation_mode="cross_scene_look_continuity")
    assert cross_scene_score.tonal_similarity >= 75.0, "Night shot should have healthy tonal score in cross-scene mode"
    assert cross_scene_score.overall_score > same_scene_score.overall_score

def test_night_preservation_does_not_blow_out_night_scene():
    # Night frame
    night_frame = np.full((100, 100, 3), 25, dtype=np.uint8)
    
    # Cross-scene grade plan (creative look without same-scene luminance forcing)
    plan = GradePlan(shot_id="night", is_same_scene=False)
    plan.creative_look.contrast = 1.10
    plan.creative_look.saturation = 1.05
    plan.creative_look.black_toe_lift = 2.0
    
    graded_night = apply_color_grade_to_frame(night_frame, plan)
    metrics_graded = compute_frame_metrics(graded_night)
    
    # Night shot median/mean must remain dark (< 45), NOT blown out to daytime 150+
    assert metrics_graded.mean_luminance < 45.0, f"Night shot was blown out! Mean: {metrics_graded.mean_luminance}"
    assert metrics_graded.p5_luminance < 30.0