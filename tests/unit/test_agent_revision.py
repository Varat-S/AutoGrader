import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.agent import AutonomousColoristAgent
from app.models.analysis import (
    SequenceInspectionResult,
    ShotSemanticAnalysis,
    CinematographyResearchResult,
    CreativeSpecification,
    SearchCitation
)
from app.models.grade import GradeResult

@pytest.fixture
def sample_video_paths():
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "sample_videos"
    ref = str(fixtures_dir / "neutral_reference.mp4")
    underexposed = str(fixtures_dir / "underexposed.mp4")
    warm = str(fixtures_dir / "warm_cast.mp4")
    return [ref, underexposed, warm]

def test_production_agent_mixed_sequence_revision_state_machine(sample_video_paths, tmp_path):
    # Mock at the service boundary (Gemini & Parallel)
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Daylight reference shot",
                lighting_environment="outdoor daylight",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.95
            ),
            ShotSemanticAnalysis(
                shot_id="shot_B",
                scene_group_id="group_1",
                relationship_to_reference="same_scene",
                scene_description="Daylight angle 2 - underexposed take",
                lighting_environment="outdoor daylight",
                time_of_day="day",
                exposure_assessment="underexposed",
                target_exposure_compensation_ev=1.5,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.60
            ),
            ShotSemanticAnalysis(
                shot_id="shot_C",
                scene_group_id="group_2",
                relationship_to_reference="independent_scene",
                scene_description="Night exterior scene with blue ambient",
                lighting_environment="night blue ambient",
                time_of_day="night",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=3.0,
                people_present=False,
                dominant_color_cast="cool / blue",
                reference_suitability_score=0.70
            )
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="mixed_sequence"
    )
    
    mock_research = CinematographyResearchResult(
        query="desert sci-fi cinematography",
        objective="research",
        sources=[SearchCitation(title="ASC", url="https://theasc.com/article", excerpt="Warm amber highlights and cool slate shadows.")],
        is_grounded=True
    )
    
    mock_spec = CreativeSpecification(
        look_title="Arrakis Desert",
        target_aesthetic="Warm golden highlights, cool slate shadows, dense blacks",
        contrast_intent=1.12,
        saturation_intent=1.05,
        highlight_bias="warm amber",
        shadow_bias="cool slate",
        black_level_treatment="filmic lifted",
        temperature_shift=3.0,
        tint_shift=-1.0,
        black_mist_diffusion_strength=0.2,
        cinematography_principles=["Highlight warmth", "Cool shadow separation"],
        citations=mock_research.sources
    )

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=sample_video_paths,
            creative_prompt="desert sci-fi look",
            job_id="test_revision_job"
        )

    assert result["reference_shot_id"] == "shot_A"
    assert os.path.exists(result["shared_lut_path"])
    assert len(result["results"]) == 3
    
    shot_a = result["results"][0]
    shot_b = result["results"][1]
    shot_c = result["results"][2]
    
    # 1. Shot A: Master Reference
    assert shot_a["state"] == "ACCEPTED"
    assert os.path.exists(shot_a["output_video_path"])
    assert os.path.exists(shot_a["lut_path"])
    
    # 2. Shot B: Same-Scene Match
    assert shot_b["state"] == "ACCEPTED"
    assert shot_b["after_consistency"]["overall_score"] >= 75.0
    assert len(shot_b["history"]) >= 1
    assert os.path.exists(shot_b["output_video_path"])
    assert os.path.exists(shot_b["lut_path"])
    # Initial evaluation was recorded
    assert shot_b["history"][0]["state"] in ["INITIAL_EVALUATION", "ACCEPTED"]
    
    # 3. Shot C: Independent Scene (Cross-Scene Look Continuity)
    # Must evaluate look continuity (not forced to daylight Lab centroid)
    assert shot_c["after_consistency"]["evaluation_mode"] == "cross_scene_look_continuity"
    assert shot_c["state"] in ["ACCEPTED", "MAX_REVISIONS_REACHED"]
    assert shot_c["after_consistency"]["overall_score"] >= 75.0
    assert os.path.exists(shot_c["output_video_path"])

def test_best_plan_retained_when_revision_is_worse(sample_video_paths, tmp_path):
    # Test that if a proposal decreases score, it is marked REVISION_REJECTED and the best plan is rendered
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
                scene_description="Target",
                lighting_environment="daylight",
                time_of_day="day",
                exposure_assessment="underexposed",
                target_exposure_compensation_ev=-1.5,
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
    mock_spec = CreativeSpecification(
        look_title="Test Look",
        target_aesthetic="Aesthetic",
        contrast_intent=1.1,
        saturation_intent=1.0,
        highlight_bias="neutral",
        shadow_bias="neutral",
        black_level_treatment="filmic",
        temperature_shift=0.0,
        tint_shift=0.0,
        black_mist_diffusion_strength=0.0,
        cinematography_principles=[],
        citations=[]
    )

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=sample_video_paths[:2],
            creative_prompt="test prompt",
            job_id="test_revert_job"
        )
        
    shot_b = result["results"][1]
    assert shot_b["plan"] is not None
    assert shot_b["state"] in ["NO_ACTIONABLE_REVISION", "MAX_REVISIONS_REACHED"]
    assert any(h["state"] == "REVISION_REJECTED" for h in shot_b["history"])
    # Verify bounds on all parameters
    assert -2.5 <= shot_b["plan"]["technical_balance"]["exposure_ev"] <= 2.5
    assert -40.0 <= shot_b["plan"]["technical_balance"]["temperature"] <= 40.0
    assert -25.0 <= shot_b["plan"]["technical_balance"]["tint"] <= 25.0