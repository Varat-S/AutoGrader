import os
import pytest
from pathlib import Path
from app.agent import AutonomousColoristAgent

@pytest.fixture
def sample_videos():
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "sample_videos"
    return [
        str(fixtures_dir / "neutral_reference.mp4"),
        str(fixtures_dir / "underexposed.mp4"),
        str(fixtures_dir / "warm_cast.mp4")
    ]

def test_full_autonomous_colorist_agent(sample_videos, tmp_path):
    agent = AutonomousColoristAgent(work_dir=str(tmp_path))
    prompt = "Restrained desert sci-fi aesthetic. Warm golden highlights, muted saturation, natural skin, cool slate shadows."
    
    result = agent.process_sequence(
        video_paths=sample_videos,
        creative_prompt=prompt,
        job_id="test_job_full"
    )
    
    # 1. Verify general job result structure
    assert result["job_id"] == "test_job_full"
    assert result["reference_shot_id"] in ["shot_A", "shot_B", "shot_C"]
    assert len(result["results"]) == 3 # All 3 shots in sequence graded (Reference + Targets)
    assert os.path.exists(result["shared_lut_path"]), "Shared creative look LUT must exist"
    
    # 2. Verify Parallel research citations (or ungrounded fallback state)
    assert "research_citations" in result
        
    # 3. Verify Creative Specification synthesized by Gemini
    spec = result["creative_specification"]
    assert spec["contrast_intent"] > 0.0
    assert spec["saturation_intent"] > 0.0
    assert len(spec["look_title"]) > 0
    
    # 4. Verify Graded Outputs and 3D LUTs
    for graded in result["results"]:
        assert os.path.exists(graded["output_video_path"]), f"Missing video: {graded['output_video_path']}"
        assert os.path.exists(graded["lut_path"]), f"Missing LUT: {graded['lut_path']}"
        print(f"\n{graded['target_shot_id']} score: {graded['before_consistency']['overall_score']} -> {graded['after_consistency']['overall_score']}")
        print(f"Explanation: {graded['explanation']}")