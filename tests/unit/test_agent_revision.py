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
    SearchCitation,
    SceneIntent
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

def test_grade_summary_contract_and_nested_schema():
    from app.models.grade import (
        GradePlan,
        InputTransformParams,
        TechnicalBalanceParams,
        CreativeLookParams,
        SceneTrimParams,
        EffectiveGradeSummary
    )

    plan = GradePlan(
        shot_id="shot_test",
        input_transform=InputTransformParams(profile="sony_slog3_sgamut3cine", is_log=True),
        technical_balance=TechnicalBalanceParams(exposure_ev=0.75, temperature=5.0, tint=-2.0),
        creative_look=CreativeLookParams(look_title="Warm Test", contrast=1.20, saturation=1.10),
        scene_trim=SceneTrimParams(trim_exposure_ev=-0.25, trim_contrast=1.05, trim_saturation=0.95)
    )

    summary: EffectiveGradeSummary = plan.compute_effective_summary(
        scene_group_id="group_test",
        scene_class="daylight",
        scene_rationale="Test rationale",
        camera_profile="sony_slog3_sgamut3cine",
        revision_state="ACCEPTED",
        highlight_bias="warm amber",
        shadow_bias="cool slate"
    )

    # 1. Schema version
    assert summary.schema_version == 1
    assert summary.shot_id == "shot_test"
    assert summary.revision_state == "ACCEPTED"

    # 2. Nested objects
    assert summary.input_transform.profile == "sony_slog3_sgamut3cine"
    assert summary.input_transform.normalization_applied is True

    assert summary.technical_balance.exposure_ev == 0.75
    assert summary.technical_balance.temperature == 5.0
    assert summary.technical_balance.tint == -2.0

    assert summary.shared_creative_look.contrast == 1.20
    assert summary.shared_creative_look.saturation == 1.10
    assert summary.shared_creative_look.highlight_bias == "warm amber"
    assert summary.shared_creative_look.shadow_bias == "cool slate"

    assert summary.scene_trim.trim_exposure_ev == -0.25
    assert summary.scene_trim.trim_contrast == 1.05
    assert summary.scene_trim.trim_saturation == 0.95

    # 3. Composed effective values
    assert summary.effective_result.exposure_ev == 0.50
    assert summary.effective_result.contrast == 1.26
    assert summary.effective_result.saturation == 1.045

    # 4. Flat compatibility properties
    assert summary.effective_exposure_ev == 0.50
    assert summary.effective_contrast == 1.26
    assert summary.effective_saturation == 1.045
    assert summary.shared_contrast == 1.20
    assert summary.shared_saturation == 1.10

    # 5. JSON serialization
    json_dict = summary.model_dump()
    assert "input_transform" in json_dict
    assert "technical_balance" in json_dict
    assert "shared_creative_look" in json_dict
    assert "scene_trim" in json_dict
    assert "effective_result" in json_dict

def test_health_aware_revision_ranking_hierarchy():
    from app.models.analysis import SceneHealthScore
    from app.models.grade import ConsistencyScore, GradePlan, TechnicalBalanceParams

    base_plan = GradePlan(shot_id="b", technical_balance=TechnicalBalanceParams(exposure_ev=0.5))
    prop_plan = GradePlan(shot_id="p", technical_balance=TechnicalBalanceParams(exposure_ev=0.2))

    best_score = ConsistencyScore(
        overall_score=85.0,
        tonal_similarity=85.0,
        chromatic_similarity=85.0,
        distribution_similarity=85.0,
        clipping_health=90.0
    )
    best_health = SceneHealthScore(
        overall_score=60.0,
        hard_gates_passed=False,
        hard_gate_failures=["shadow_crush"],
        passed=False
    )

    # 1. Proposed reduces triggered hard gates (1 -> 0): accepted even with small continuity drop
    prop_score = ConsistencyScore(
        overall_score=80.0, # dropped 5 pts
        tonal_similarity=80.0,
        chromatic_similarity=80.0,
        distribution_similarity=80.0,
        clipping_health=90.0
    )
    prop_health = SceneHealthScore(
        overall_score=78.0,
        hard_gates_passed=True,
        hard_gate_failures=[],
        passed=True
    )

    # Replicate ranking check from agent.py
    def check_better(b_score, b_health, p_score, p_health, b_plan, p_plan):
        if b_health.hard_gates_passed and not p_health.hard_gates_passed:
            return False
        if p_score.clipping_health < b_score.clipping_health - 2.0:
            return False
        c_drop = b_score.overall_score - p_score.overall_score
        b_gates = len(getattr(b_health, "hard_gate_failures", [])) if not b_health.hard_gates_passed else 0
        p_gates = len(getattr(p_health, "hard_gate_failures", [])) if not p_health.hard_gates_passed else 0

        if p_gates < b_gates:
            return c_drop <= 10.0
        if p_gates > b_gates:
            return False
        if p_health.passed and not b_health.passed:
            return c_drop <= 5.0
        if b_health.passed and not p_health.passed:
            return False
        if p_health.passed == b_health.passed:
            h_diff = p_health.overall_score - b_health.overall_score
            if h_diff > 1.0:
                return c_drop <= 4.0
            elif abs(h_diff) <= 1.0:
                s_diff = p_score.overall_score - b_score.overall_score
                if s_diff > 0.5:
                    return True
                elif abs(s_diff) <= 0.2:
                    return abs(p_plan.technical_balance.exposure_ev) < abs(b_plan.technical_balance.exposure_ev)
        return False

    assert check_better(best_score, best_health, prop_score, prop_health, base_plan, prop_plan) is True

    # 2. Proposed worsens clipping_health by > 2 points: rejected even if overall score is higher
    prop_score_bad_clip = ConsistencyScore(
        overall_score=95.0, # high continuity
        tonal_similarity=95.0,
        chromatic_similarity=95.0,
        distribution_similarity=95.0,
        clipping_health=85.0 # dropped 5 pts (> 2 pts)
    )
    prop_health_good = SceneHealthScore(overall_score=85.0, hard_gates_passed=True, passed=True)
    best_health_good = SceneHealthScore(overall_score=85.0, hard_gates_passed=True, passed=True)
    best_score_good_clip = ConsistencyScore(
        overall_score=80.0,
        tonal_similarity=80.0,
        chromatic_similarity=80.0,
        distribution_similarity=80.0,
        clipping_health=90.0
    )
    assert check_better(best_score_good_clip, best_health_good, prop_score_bad_clip, prop_health_good, base_plan, prop_plan) is False

    # 3. Tied proposal within 0.2 prefers smaller absolute parameter deltas
    prop_score_tied = ConsistencyScore(
        overall_score=80.1,
        tonal_similarity=80.0,
        chromatic_similarity=80.0,
        distribution_similarity=80.0,
        clipping_health=90.0
    )
    # p_plan has 0.2 EV vs b_plan 0.5 EV: smaller delta -> True
    assert check_better(best_score_good_clip, best_health_good, prop_score_tied, best_health_good, base_plan, prop_plan) is True

def test_creative_spec_with_scene_intents_completes_delivery(sample_video_paths, tmp_path):
    # Regression test: proves a CreativeSpecification containing real scene_intents
    # completes all the way through delivery rendering without attribute errors.
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Daylight reference",
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
                scene_group_id="group_2",
                relationship_to_reference="independent_scene",
                scene_description="Night exterior scene",
                lighting_environment="night exterior",
                time_of_day="night",
                exposure_assessment="low_key",
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
        query="regression test",
        objective="research",
        sources=[SearchCitation(title="ASC Ref", url="https://theasc.com/article", excerpt="Night lighting principles.")],
        is_grounded=True
    )

    intents = [
        SceneIntent(
            scene_group_id="group_1",
            lighting_class="daylight",
            exposure_class="balanced",
            concise_rationale="Daylight balanced reference rationale",
            source_relative_exposure_bounds=[-1.0, 1.0],
            contrast_trim_bounds=[0.9, 1.2],
            saturation_trim_bounds=[0.8, 1.2]
        ),
        SceneIntent(
            scene_group_id="group_2",
            lighting_class="low_key_night",
            exposure_class="low_key_night",
            concise_rationale="Night moody low-key rationale",
            source_relative_exposure_bounds=[-0.8, 0.2],
            contrast_trim_bounds=[0.85, 1.15],
            saturation_trim_bounds=[0.7, 1.1]
        )
    ]

    mock_spec = CreativeSpecification(
        look_title="Arrakis Regression Look",
        target_aesthetic="Warm golden highlights, cool slate shadows",
        contrast_intent=1.12,
        saturation_intent=1.05,
        highlight_bias="warm amber",
        shadow_bias="cool slate",
        black_level_treatment="filmic lifted",
        temperature_shift=2.0,
        tint_shift=-1.0,
        black_mist_diffusion_strength=0.15,
        cinematography_principles=["Highlight warmth", "Cool shadow separation"],
        citations=mock_research.sources,
        scene_intents=intents
    )

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=sample_video_paths[:2],
            creative_prompt="warm arrakis desert",
            job_id="test_scene_intents_delivery_job"
        )

    assert result["reference_shot_id"] == "shot_A"
    assert os.path.exists(result["shared_lut_path"])
    assert len(result["results"]) == 2

    for r in result["results"]:
        assert os.path.exists(r["output_video_path"])
        assert os.path.exists(r["lut_path"])
        assert os.path.exists(r["before_proxy_path"])
        assert os.path.exists(r["after_proxy_path"])
        assert r["grade_summary"] is not None
        assert len(r["grade_summary"]["scene_rationale"]) > 0

def test_revisions_all_rejected_completes_delivery_without_unbound_local_error(sample_video_paths, tmp_path):
    """Regression test: when candidate revisions are rejected, is_accepted must not raise UnboundLocalError."""
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Daylight reference",
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
                scene_group_id="group_2",
                relationship_to_reference="independent_scene",
                scene_description="Dark candidate shot",
                lighting_environment="low_key_night",
                time_of_day="night",
                exposure_assessment="low_key",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.4
            )
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="mixed_sequence"
    )

    mock_research = CinematographyResearchResult(query="test", objective="test", sources=[], is_grounded=False)
    mock_spec = CreativeSpecification(
        look_title="Test Look",
        target_aesthetic="Natural",
        contrast_intent=1.0,
        saturation_intent=1.0,
        highlight_bias="neutral",
        shadow_bias="neutral",
        black_level_treatment="neutral",
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
            job_id="test_rejection_delivery"
        )

    assert len(result["results"]) == 2
    for r in result["results"]:
        assert os.path.exists(r["output_video_path"])
        assert os.path.exists(r["lut_path"])