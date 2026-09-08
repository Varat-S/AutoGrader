import pytest
import numpy as np
from unittest.mock import patch
from app.models.analysis import (
    ShotMetrics,
    SequenceInspectionResult,
    ShotSemanticAnalysis,
    CinematographyResearchResult,
    CreativeSpecification,
    NormalizationValidationResult
)
from app.media.color import assess_normalization_health, aggregate_shot_metrics
from app.agent import AutonomousColoristAgent

def test_normalization_assessment_unit_states():
    # 1. Flat footage under Rec.709 triggers confirmation required
    flat_frame = np.full((50, 50, 3), 42, dtype=np.uint8)
    metrics_flat = aggregate_shot_metrics("flat", "flat.mp4", [flat_frame], [0.0], 30.0, 50, 50, 1.0)
    res_rec = assess_normalization_health("flat", metrics_flat, [flat_frame], profile="rec709")
    assert res_rec.state == "PROFILE_CONFIRMATION_REQUIRED"
    assert not res_rec.passed

    # 2. Overclipped frame triggers NORMALIZATION_FAILED
    clipped_frame = np.zeros((50, 50, 3), dtype=np.uint8) # 100% shadow clip
    metrics_clip = aggregate_shot_metrics("clip", "clip.mp4", [clipped_frame], [0.0], 30.0, 50, 50, 1.0)
    res_clip = assess_normalization_health("clip", metrics_clip, [clipped_frame], profile="rec709")
    assert res_clip.state == "NORMALIZATION_FAILED"
    assert not res_clip.passed

    # 3. Healthy frame passes verification
    healthy_frame = np.full((50, 50, 3), 120, dtype=np.uint8)
    healthy_frame[:10, :] = 10
    healthy_frame[-10:, :] = 220
    metrics_healthy = aggregate_shot_metrics("healthy", "healthy.mp4", [healthy_frame], [0.0], 30.0, 50, 50, 1.0)
    res_healthy = assess_normalization_health("healthy", metrics_healthy, [healthy_frame], profile="rec709")
    assert res_healthy.state == "NORMALIZATION_VERIFIED"
    assert res_healthy.passed

def test_master_reference_normalization_gate_blocks_agent(tmp_path):
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Ref",
                lighting_environment="daylight",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.9
            )
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="continuous_sequence"
    )
    mock_research = CinematographyResearchResult(query="q", objective="o", sources=[], is_grounded=False)
    mock_spec = CreativeSpecification(look_title="Test", target_aesthetic="Aesthetic", contrast_intent=1.0, saturation_intent=1.0, highlight_bias="neutral", shadow_bias="neutral", black_level_treatment="neutral", temperature_shift=0.0, tint_shift=0.0, black_mist_diffusion_strength=0.0, cinematography_principles=[], citations=[])

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    failed_norm = NormalizationValidationResult(
        shot_id="shot_A",
        state="PROFILE_CONFIRMATION_REQUIRED",
        passed=False,
        reason="Elevated black floor indicates Log footage under Rec.709."
    )

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec), \
         patch("app.agent.assess_normalization_health", return_value=failed_norm):

        with pytest.raises(RuntimeError, match="requires profile confirmation"):
            agent.process_sequence(
                video_paths=["tests/fixtures/sample_videos/neutral_reference.mp4"],
                creative_prompt="test prompt",
                color_profile="rec709",
                input_profiles=[{"shot_index": 0, "profile": "rec709", "user_confirmed": False, "override_warning": False}]
            )

def test_normalization_gate_override_allows_continuation(tmp_path):
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Ref",
                lighting_environment="daylight",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.9
            )
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="continuous_sequence"
    )
    mock_research = CinematographyResearchResult(query="q", objective="o", sources=[], is_grounded=False)
    mock_spec = CreativeSpecification(look_title="Test", target_aesthetic="Aesthetic", contrast_intent=1.0, saturation_intent=1.0, highlight_bias="neutral", shadow_bias="neutral", black_level_treatment="neutral", temperature_shift=0.0, tint_shift=0.0, black_mist_diffusion_strength=0.0, cinematography_principles=[], citations=[])

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    warning_norm = NormalizationValidationResult(
        shot_id="shot_A",
        state="PROFILE_CONFIRMATION_REQUIRED",
        passed=False,
        reason="Elevated black floor indicates Log footage under Rec.709."
    )

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec), \
         patch("app.agent.assess_normalization_health", return_value=warning_norm):

        res = agent.process_sequence(
            video_paths=["tests/fixtures/sample_videos/neutral_reference.mp4"],
            creative_prompt="test prompt",
            color_profile="rec709",
            input_profiles=[{"shot_index": 0, "profile": "rec709", "user_confirmed": True, "override_warning": True}]
        )
        assert len(res["results"]) == 1
        assert res["results"][0]["target_shot_id"] == "shot_A"
        assert res["normalization_results"][0]["state"] == "NORMALIZATION_WARNING_OVERRIDDEN"

def test_candidate_normalization_failure_halts_agent(tmp_path):
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Ref",
                lighting_environment="daylight",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.9
            ),
            ShotSemanticAnalysis(
                shot_id="shot_B",
                scene_group_id="group_1",
                relationship_to_reference="same_scene",
                scene_description="Cand",
                lighting_environment="daylight",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.5
            )
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="continuous_sequence"
    )
    mock_research = CinematographyResearchResult(query="q", objective="o", sources=[], is_grounded=False)
    mock_spec = CreativeSpecification(look_title="Test", target_aesthetic="Aesthetic", contrast_intent=1.0, saturation_intent=1.0, highlight_bias="neutral", shadow_bias="neutral", black_level_treatment="neutral", temperature_shift=0.0, tint_shift=0.0, black_mist_diffusion_strength=0.0, cinematography_principles=[], citations=[])

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    def mock_norm_side_effect(shot_id, *args, **kwargs):
        if shot_id == "shot_A":
            return NormalizationValidationResult(shot_id="shot_A", state="NORMALIZATION_VERIFIED", passed=True, reason="OK")
        else:
            return NormalizationValidationResult(shot_id="shot_B", state="NORMALIZATION_FAILED", passed=False, reason="Excessive clipping")

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec), \
         patch("app.agent.assess_normalization_health", side_effect=mock_norm_side_effect):

        with pytest.raises(RuntimeError, match="failed normalization gate"):
            agent.process_sequence(
                video_paths=["tests/fixtures/sample_videos/neutral_reference.mp4", "tests/fixtures/sample_videos/underexposed.mp4"],
                creative_prompt="test prompt",
                color_profile="rec709"
            )