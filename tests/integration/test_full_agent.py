import os
import pytest
from pathlib import Path
from unittest.mock import patch

from app.agent import AutonomousColoristAgent
from app.models.analysis import (
    SequenceInspectionResult,
    ShotSemanticAnalysis,
    CinematographyResearchResult,
    CreativeSpecification,
    SearchCitation
)

@pytest.fixture
def sample_videos():
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "sample_videos"
    return [
        str(fixtures_dir / "neutral_reference.mp4"),
        str(fixtures_dir / "underexposed.mp4"),
        str(fixtures_dir / "warm_cast.mp4")
    ]

def test_full_autonomous_colorist_agent_deterministic(sample_videos, tmp_path):
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
                scene_description="Daylight underexposed shot",
                lighting_environment="outdoor daylight",
                time_of_day="day",
                exposure_assessment="underexposed",
                target_exposure_compensation_ev=1.2,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.60
            ),
            ShotSemanticAnalysis(
                shot_id="shot_C",
                scene_group_id="group_2",
                relationship_to_reference="independent_scene",
                scene_description="Warm cast independent scene",
                lighting_environment="golden hour",
                time_of_day="golden_hour",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.5,
                people_present=False,
                dominant_color_cast="warm / golden",
                reference_suitability_score=0.75
            )
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="mixed_sequence"
    )
    
    mock_research = CinematographyResearchResult(
        query="desert sci-fi",
        objective="research",
        sources=[SearchCitation(title="ASC", url="https://theasc.com/article", excerpt="Warm highlights, cool slate shadows.")],
        is_grounded=True
    )
    
    mock_spec = CreativeSpecification(
        look_title="Desert Sci-Fi",
        target_aesthetic="Warm golden highlights, muted saturation, cool slate shadows",
        contrast_intent=1.12,
        saturation_intent=1.05,
        highlight_bias="warm golden",
        shadow_bias="cool slate",
        black_level_treatment="filmic lifted",
        temperature_shift=3.0,
        tint_shift=-1.0,
        black_mist_diffusion_strength=0.2,
        cinematography_principles=["Highlight roll-off", "Complementary color separation"],
        citations=mock_research.sources
    )

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=sample_videos,
            creative_prompt="Restrained desert sci-fi aesthetic.",
            job_id="test_job_det"
        )
        
    assert result["job_id"] == "test_job_det"
    assert result["reference_shot_id"] == "shot_A"
    assert len(result["results"]) == 3
    assert os.path.exists(result["shared_lut_path"])
    
    for r in result["results"]:
        assert os.path.exists(r["output_video_path"])
        assert os.path.exists(r["lut_path"])

@pytest.mark.live
def test_full_autonomous_colorist_agent_live(sample_videos, tmp_path):
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set for live test")
        
    agent = AutonomousColoristAgent(work_dir=str(tmp_path))
    prompt = "Restrained desert sci-fi aesthetic. Warm golden highlights, muted saturation, natural skin, cool slate shadows."
    
    result = agent.process_sequence(
        video_paths=sample_videos,
        creative_prompt=prompt,
        job_id="test_job_live"
    )
    
    assert len(result["results"]) == 3
    assert os.path.exists(result["shared_lut_path"])