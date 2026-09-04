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

def test_identical_look_on_differently_colored_day_night_scenes():
    # Day reference: warm daylight colors
    day_frame = np.full((100, 100, 3), [130, 160, 190], dtype=np.uint8)
    # Night candidate: deep blue ambient colors with practical
    night_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    night_frame[:80, :] = [35, 20, 15] # Deep blue/dark shadows
    night_frame[80:, :] = [70, 90, 120] # Light practical
    
    shot_day = aggregate_shot_metrics("day", "day.mp4", [day_frame], [0.0], 30.0, 100, 100, 1.0)
    shot_night = aggregate_shot_metrics("night", "night.mp4", [night_frame], [0.0], 30.0, 100, 100, 1.0)
    
    # Same unified creative look applied to both
    plan_ref = GradePlan(shot_id="day", is_same_scene=False)
    plan_ref.creative_look.contrast = 1.12
    plan_ref.creative_look.highlight_rgb_offset = [-0.04, 0.01, 0.05] # warm amber
    plan_ref.creative_look.shadow_rgb_offset = [0.05, 0.01, -0.03] # cool slate
    
    plan_night = GradePlan(shot_id="night", is_same_scene=False)
    plan_night.creative_look.contrast = 1.12
    plan_night.creative_look.highlight_rgb_offset = [-0.04, 0.01, 0.05]
    plan_night.creative_look.shadow_rgb_offset = [0.05, 0.01, -0.03]
    
    graded_night = apply_color_grade_to_frame(night_frame, plan_night)
    shot_graded_night = aggregate_shot_metrics("graded_night", "night.mp4", [graded_night], [0.0], 30.0, 100, 100, 1.0)
    
    # In cross-scene mode with standardized probes, identical look passes cleanly
    score = compute_consistency_score(
        reference=shot_day,
        candidate=shot_graded_night,
        evaluation_mode="cross_scene_look_continuity",
        ref_plan=plan_ref,
        cand_plan=plan_night
    )
    
    assert score.overall_score >= 80.0, f"Identical look should pass cross-scene continuity, got {score.overall_score}"
    assert score.chromatic_similarity >= 95.0, f"Chromatic split-tone harmony should be high on probes, got {score.chromatic_similarity}"

def test_divergent_split_bias_reduces_look_continuity():
    shot = aggregate_shot_metrics("s", "s.mp4", [np.full((50, 50, 3), 100, dtype=np.uint8)], [0.0], 30.0, 50, 50, 1.0)
    
    plan_ref = GradePlan(shot_id="ref")
    plan_ref.creative_look.highlight_rgb_offset = [-0.05, 0.01, 0.06] # warm amber
    plan_ref.creative_look.shadow_rgb_offset = [0.06, 0.0, -0.04] # cool slate
    
    # Divergent look: cool cyan highlights & warm brown shadows
    plan_divergent = GradePlan(shot_id="div")
    plan_divergent.creative_look.highlight_rgb_offset = [0.06, 0.01, -0.05] # cool cyan
    plan_divergent.creative_look.shadow_rgb_offset = [-0.05, 0.0, 0.05] # warm brown
    
    score = compute_consistency_score(
        reference=shot,
        candidate=shot,
        evaluation_mode="cross_scene_look_continuity",
        ref_plan=plan_ref,
        cand_plan=plan_divergent
    )
    
    assert score.chromatic_similarity < 50.0, f"Divergent highlight/shadow biases must drop chromatic similarity, got {score.chromatic_similarity}"

def test_night_preservation_does_not_blow_out_night_scene():
    night_frame = np.full((100, 100, 3), 25, dtype=np.uint8)
    
    plan = GradePlan(shot_id="night", is_same_scene=False)
    plan.creative_look.contrast = 1.10
    plan.creative_look.saturation = 1.05
    plan.creative_look.black_toe_lift = 2.0
    
    graded_night = apply_color_grade_to_frame(night_frame, plan)
    metrics_graded = compute_frame_metrics(graded_night)
    
    assert metrics_graded.mean_luminance < 45.0, f"Night shot was blown out! Mean: {metrics_graded.mean_luminance}"
    assert metrics_graded.p5_luminance < 30.0

def test_saturation_mismatch_detected_by_probes():
    shot = aggregate_shot_metrics("s", "s.mp4", [np.full((50, 50, 3), 100, dtype=np.uint8)], [0.0], 30.0, 50, 50, 1.0)
    
    plan_ref = GradePlan(shot_id="ref")
    plan_ref.creative_look.saturation = 1.0
    
    plan_oversat = GradePlan(shot_id="oversat")
    plan_oversat.creative_look.saturation = 2.5
    
    score = compute_consistency_score(
        reference=shot,
        candidate=shot,
        evaluation_mode="cross_scene_look_continuity",
        ref_plan=plan_ref,
        cand_plan=plan_oversat
    )
    assert score.distribution_similarity < 60.0 or score.chromatic_similarity < 60.0
    assert score.overall_score < 75.0

def test_contrast_mismatch_detected_by_probes():
    shot = aggregate_shot_metrics("s", "s.mp4", [np.full((50, 50, 3), 100, dtype=np.uint8)], [0.0], 30.0, 50, 50, 1.0)
    
    plan_low_contrast = GradePlan(shot_id="low")
    plan_low_contrast.creative_look.contrast = 0.5
    
    plan_high_contrast = GradePlan(shot_id="high")
    plan_high_contrast.creative_look.contrast = 2.0
    
    score = compute_consistency_score(
        reference=shot,
        candidate=shot,
        evaluation_mode="cross_scene_look_continuity",
        ref_plan=plan_low_contrast,
        cand_plan=plan_high_contrast
    )
    assert score.tonal_similarity <= 45.0
    assert score.overall_score < 60.0

def test_day_night_trims_do_not_contaminate_creative_continuity():
    shot_day = aggregate_shot_metrics("day", "day.mp4", [np.full((50, 50, 3), 120, dtype=np.uint8)], [0.0], 30.0, 50, 50, 1.0)
    shot_night = aggregate_shot_metrics("night", "night.mp4", [np.full((50, 50, 3), 35, dtype=np.uint8)], [0.0], 30.0, 50, 50, 1.0)
    
    # Identical creative look
    plan_ref = GradePlan(shot_id="day")
    plan_ref.creative_look.contrast = 1.15
    plan_ref.creative_look.saturation = 1.05
    plan_ref.creative_look.highlight_rgb_offset = [-0.04, 0.01, 0.05]
    plan_ref.creative_look.shadow_rgb_offset = [0.05, 0.01, -0.03]
    
    plan_night = GradePlan(shot_id="night")
    plan_night.creative_look.contrast = 1.15
    plan_night.creative_look.saturation = 1.05
    plan_night.creative_look.highlight_rgb_offset = [-0.04, 0.01, 0.05]
    plan_night.creative_look.shadow_rgb_offset = [0.05, 0.01, -0.03]
    # Night shot has distinct technical balance and scene trim
    plan_night.technical_balance.exposure_ev = -1.2
    plan_night.scene_trim.trim_exposure_ev = -0.5
    
    score = compute_consistency_score(
        reference=shot_day,
        candidate=shot_night,
        evaluation_mode="cross_scene_look_continuity",
        ref_plan=plan_ref,
        cand_plan=plan_night
    )
    assert score.overall_score >= 80.0
    assert score.tonal_similarity >= 80.0
    assert score.chromatic_similarity >= 80.0