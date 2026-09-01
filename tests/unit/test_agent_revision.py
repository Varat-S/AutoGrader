import pytest
import numpy as np
from pathlib import Path
from app.models.analysis import ShotMetrics, ShotSemanticAnalysis, CreativeSpecification
from app.models.grade import GradePlan, ConsistencyScore
from app.media.color import aggregate_shot_metrics, apply_color_grade_to_frame, compute_consistency_score
from app.tools.calculate_grade import build_grade_plan

def test_autonomous_diagnostic_revision_loop():
    # Setup daylight reference
    ref_frame = np.full((50, 50, 3), [130, 140, 150], dtype=np.uint8)
    ref_metrics = aggregate_shot_metrics("shot_A", "ref.mp4", [ref_frame], [0.0], 30.0, 50, 50, 1.0)
    
    # Setup underexposed candidate shot
    tgt_frame = np.full((50, 50, 3), [40, 45, 50], dtype=np.uint8)
    tgt_metrics = aggregate_shot_metrics("shot_B", "tgt.mp4", [tgt_frame], [0.0], 30.0, 50, 50, 1.0)
    
    spec = CreativeSpecification(
        look_title="Test Look",
        target_aesthetic="Film look",
        contrast_intent=1.1,
        saturation_intent=1.05
    )
    
    # Grade Reference
    ref_plan = build_grade_plan(ref_metrics, ref_metrics, creative_spec=spec, is_reference_shot=True)
    graded_ref_frame = apply_color_grade_to_frame(ref_frame, ref_plan)
    graded_ref_metrics = aggregate_shot_metrics("graded_ref", "ref.mp4", [graded_ref_frame], [0.0], 30.0, 50, 50, 1.0)
    
    # Initial Target Plan
    target_plan = build_grade_plan(ref_metrics, tgt_metrics, creative_spec=spec, is_same_scene=True)
    initial_ev = target_plan.technical_balance.exposure_ev
    
    revisions_performed = 0
    for rev in range(2):
        preview_frame = apply_color_grade_to_frame(tgt_frame, target_plan)
        cand_metrics = aggregate_shot_metrics("cand", "cand.mp4", [preview_frame], [0.0], 30.0, 50, 50, 1.0)
        score = compute_consistency_score(graded_ref_metrics, cand_metrics, evaluation_mode="same_scene_match")
        
        if score.overall_score >= 75.0:
            break
            
        revisions_performed += 1
        # Diagnostic EV trim
        p50_target = graded_ref_metrics.p50_luminance if graded_ref_metrics.p50_luminance > 0 else graded_ref_metrics.avg_luminance
        p50_cand = cand_metrics.p50_luminance if cand_metrics.p50_luminance > 0 else cand_metrics.avg_luminance
        ev_adj = float(np.log2(max(1.0, p50_target) / max(1.0, p50_cand)) * 0.45)
        target_plan.technical_balance.exposure_ev += round(ev_adj, 2)
        
    assert revisions_performed >= 1, "Should have performed at least 1 diagnostic revision"
    assert target_plan.technical_balance.exposure_ev > initial_ev, "Revised exposure must be higher than initial"