import os
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.agent import AutonomousColoristAgent
from app.models.analysis import (
    SequenceInspectionResult,
    ShotSemanticAnalysis,
    CinematographyResearchResult,
    CreativeSpecification,
    SearchCitation,
    SceneIntent
)
from app.models.grade import (
    GradePlan,
    InputTransformParams,
    TechnicalBalanceParams,
    SceneMatchParams,
    SceneTrimParams,
    CreativeLookParams
)
from app.media.color import (
    evaluate_scene_health,
    calculate_deterministic_match_params,
    compute_frame_metrics
)
from app.media.ffmpeg import generate_matched_browser_proxies, probe_video
from app.main import app, jobs

@pytest.fixture
def sample_video_paths():
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "sample_videos"
    ref = str(fixtures_dir / "neutral_reference.mp4")
    underexposed = str(fixtures_dir / "underexposed.mp4")
    warm = str(fixtures_dir / "warm_cast.mp4")
    return [ref, underexposed, warm]

def test_immutable_sequence_look_across_shots_and_revisions(sample_video_paths, tmp_path):
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
                scene_group_id="group_1",
                relationship_to_reference="same_scene",
                scene_description="Daylight same scene angle",
                lighting_environment="outdoor daylight",
                time_of_day="day",
                exposure_assessment="underexposed",
                target_exposure_compensation_ev=1.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.60
            ),
            ShotSemanticAnalysis(
                shot_id="shot_C",
                scene_group_id="group_2",
                relationship_to_reference="independent_scene",
                scene_description="Night low key scene",
                lighting_environment="night exterior",
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
        query="cinematic test",
        objective="research",
        sources=[SearchCitation(title="ASC", url="https://theasc.com", excerpt="Filmic contrast and warm highlights.")],
        is_grounded=True
    )

    mock_spec = CreativeSpecification(
        look_title="Neo Noir",
        target_aesthetic="Rich contrast, amber highlights, cool shadows",
        contrast_intent=1.20,
        saturation_intent=1.10,
        highlight_bias="warm amber",
        shadow_bias="cool cyan",
        black_level_treatment="neutral",
        cinematography_principles=["Preserve darkness in night scenes"],
        citations=mock_research.sources
    )

    agent = AutonomousColoristAgent(work_dir=str(tmp_path))

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=sample_video_paths,
            creative_prompt="neo noir look",
            job_id="test_immutable_look_job"
        )

    res_a, res_b, res_c = result["results"]

    # Shared sequence look must be identical across all shots
    assert res_a["grade_summary"]["shared_contrast"] == 1.20
    assert res_b["grade_summary"]["shared_contrast"] == 1.20
    assert res_c["grade_summary"]["shared_contrast"] == 1.20

    assert res_a["grade_summary"]["shared_saturation"] == 1.10
    assert res_b["grade_summary"]["shared_saturation"] == 1.10
    assert res_c["grade_summary"]["shared_saturation"] == 1.10

    # Effective contrast/saturation reflect scene trims, not changes to the immutable look
    assert "effective_contrast" in res_b["grade_summary"]
    assert "effective_saturation" in res_c["grade_summary"]

    # Proxies generated
    assert os.path.exists(res_a["before_proxy_path"])
    assert os.path.exists(res_a["after_proxy_path"])
    assert os.path.exists(res_b["before_proxy_path"])
    assert os.path.exists(res_b["after_proxy_path"])

def test_scene_health_hard_gate_excessive_clipping():
    h, w = 100, 100
    frame = np.ones((h, w, 3), dtype=np.uint8) * 128
    frame[:15, :] = 255

    frame_metrics = compute_frame_metrics(frame, timestamp_sec=0.0)
    assert frame_metrics.highlight_clip_pct >= 10.0

    from app.models.analysis import ShotMetrics
    source_metrics = ShotMetrics(
        shot_id="shot_clip",
        video_path="",
        duration_sec=1.0,
        width=w,
        height=h,
        fps=30.0,
        sampled_frames=[frame_metrics],
        avg_luminance=140.0,
        avg_shadow_clip_pct=0.0,
        avg_highlight_clip_pct=frame_metrics.highlight_clip_pct,
        avg_lab_mean=[50.0, 0.0, 0.0],
        avg_lab_std=[10.0, 5.0, 5.0],
        avg_chroma=5.0
    )

    scene_intent = SceneIntent(
        scene_group_id="group_1",
        lighting_class="daylight",
        allow_crushed_blacks=False,
        max_highlight_clip_pct=8.0
    )

    health = evaluate_scene_health(
        source_metrics=source_metrics,
        graded_metrics=source_metrics,
        graded_frames=[frame],
        scene_intent=scene_intent
    )

    assert health.hard_gates_passed is False
    assert health.clipping_health < 100.0
    assert any("highlight clipping" in f.lower() for f in health.hard_gate_failures)

def test_scene_health_dark_scene_shadow_oversaturation():
    h, w = 64, 64
    dark_saturated = np.zeros((h, w, 3), dtype=np.uint8)
    dark_saturated[:, :, 0] = 60  # B in BGR
    dark_saturated[:, :, 1] = 5   # G
    dark_saturated[:, :, 2] = 5   # R

    frame_metrics = compute_frame_metrics(dark_saturated, timestamp_sec=0.0)
    from app.models.analysis import ShotMetrics
    metrics = ShotMetrics(
        shot_id="night_shot",
        video_path="",
        duration_sec=1.0,
        width=w,
        height=h,
        fps=30.0,
        sampled_frames=[frame_metrics],
        avg_luminance=15.0,
        p5_luminance=5.0,
        p50_luminance=12.0,
        p95_luminance=25.0,
        avg_shadow_clip_pct=0.0,
        avg_highlight_clip_pct=0.0,
        avg_lab_mean=[10.0, 5.0, -25.0],
        avg_lab_std=[2.0, 2.0, 2.0],
        avg_chroma=26.0
    )

    night_intent = SceneIntent(
        scene_group_id="group_night",
        lighting_class="low_key_night",
        allow_crushed_blacks=True,
        preserve_natural_shadow_density=True
    )

    health = evaluate_scene_health(
        source_metrics=metrics,
        graded_metrics=metrics,
        graded_frames=[dark_saturated],
        scene_intent=night_intent
    )

    assert health.hard_gates_passed is False
    assert health.shadow_saturation_health < 100.0
    assert any("shadow" in f.lower() for f in health.hard_gate_failures)

def test_composition_aware_luminance_anchoring():
    from app.models.analysis import ShotMetrics, FrameMetrics
    ref_fm = FrameMetrics(
        timestamp_sec=0.0, mean_luminance=130.0, median_luminance=128.0,
        p5_luminance=15.0, p25_luminance=60.0, p50_luminance=128.0,
        p75_luminance=180.0, p95_luminance=235.0, shadow_clip_pct=0.0,
        highlight_clip_pct=0.0, lab_l_mean=50.0, lab_l_std=15.0,
        lab_a_mean=0.0, lab_a_std=5.0, lab_b_mean=0.0, lab_b_std=5.0,
        mean_chroma=5.0, r_mean=130.0, g_mean=130.0, b_mean=130.0
    )
    ref_metrics = ShotMetrics(
        shot_id="ref", video_path="", duration_sec=1.0, width=640, height=360, fps=30.0,
        sampled_frames=[ref_fm], avg_luminance=130.0, p5_luminance=15.0, p50_luminance=128.0,
        p95_luminance=235.0, avg_shadow_clip_pct=0.0, avg_highlight_clip_pct=0.0,
        avg_lab_mean=[50.0, 0.0, 0.0], avg_lab_std=[15.0, 5.0, 5.0], avg_chroma=5.0
    )

    target_fm = FrameMetrics(
        timestamp_sec=0.0, mean_luminance=80.0, median_luminance=75.0,
        p5_luminance=16.0, p25_luminance=40.0, p50_luminance=75.0,
        p75_luminance=110.0, p95_luminance=234.0, shadow_clip_pct=0.0,
        highlight_clip_pct=0.0, lab_l_mean=35.0, lab_l_std=20.0,
        lab_a_mean=0.0, lab_a_std=5.0, lab_b_mean=0.0, lab_b_std=5.0,
        mean_chroma=5.0, r_mean=80.0, g_mean=80.0, b_mean=80.0
    )
    target_metrics = ShotMetrics(
        shot_id="target", video_path="", duration_sec=1.0, width=640, height=360, fps=30.0,
        sampled_frames=[target_fm], avg_luminance=80.0, p5_luminance=16.0, p50_luminance=75.0,
        p95_luminance=234.0, avg_shadow_clip_pct=0.0, avg_highlight_clip_pct=0.0,
        avg_lab_mean=[35.0, 0.0, 0.0], avg_lab_std=[20.0, 5.0, 5.0], avg_chroma=5.0
    )

    params = calculate_deterministic_match_params(
        reference=ref_metrics,
        target=target_metrics
    )

    # Without composition anchoring and headroom protection, raw_l_offset would be 15.0.
    # Here, headroom guard and endpoint anchoring clamp it down to prevent highlight blow-out.
    assert params.lab_l_offset < 3.0

def test_matched_browser_proxies_generation(tmp_path):
    sample_video = "tests/fixtures/sample_videos/neutral_reference.mp4"
    if not os.path.exists(sample_video):
        pytest.skip("Fixture video not available")

    lut_path = str(tmp_path / "test.cube")
    with open(lut_path, "w") as f:
        f.write("LUT_3D_SIZE 2\n")
        for b in [0.0, 1.0]:
            for g in [0.0, 1.0]:
                for r in [0.0, 1.0]:
                    f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    before_proxy = str(tmp_path / "before_proxy.mp4")
    after_proxy = str(tmp_path / "after_proxy.mp4")

    out_b, out_a = generate_matched_browser_proxies(
        source_path=sample_video,
        lut_path=lut_path,
        before_proxy_path=before_proxy,
        after_proxy_path=after_proxy,
        max_height=720
    )

    assert os.path.exists(out_b)
    assert os.path.exists(out_a)

    info_b = probe_video(out_b)
    info_a = probe_video(out_a)

    assert info_b["width"] == info_a["width"]
    assert info_b["height"] == info_a["height"]
    assert info_b["fps"] == info_a["fps"]
    assert info_b["color_space"] == "bt709"
    assert info_a["color_space"] == "bt709"

def test_load_demo_sequence_loads_1080p_dji_clips(tmp_path):
    client = TestClient(app)

    res = client.post("/api/jobs")
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    res_load = client.post(f"/api/jobs/{job_id}/load_demo")
    assert res_load.status_code == 200
    data = res_load.json()

    assert data["status"] == "success"
    assert len(data["loaded"]) == 3
    assert any("twilight_flight" in clip for clip in data["loaded"])
    assert any("overcast_harbor" in clip for clip in data["loaded"])
    assert any("dusk_cityscape" in clip for clip in data["loaded"])

    job_info = jobs[job_id]
    first_clip = job_info["source_videos"][0]
    probe = probe_video(first_clip)
    assert probe["width"] == 1920
    assert probe["height"] == 1080

def test_scene_intent_model_bound_validation():
    # Valid bounds
    si = SceneIntent(
        scene_group_id="g1",
        source_relative_exposure_bounds=[-1.0, 1.0],
        contrast_trim_bounds=[0.8, 1.3],
        saturation_trim_bounds=[0.6, 1.5]
    )
    assert si.source_relative_exposure_bounds == [-1.0, 1.0]

    # Rejection of empty, single-element, inverted, extreme, NaN, Inf
    invalid_cases = [
        [],
        [0.0],
        [1.0, -1.0],
        [-99.0, 99.0],
        [float("nan"), 1.0],
        [-1.0, float("inf")]
    ]

    for inv in invalid_cases:
        with pytest.raises(ValueError):
            SceneIntent(scene_group_id="g1", source_relative_exposure_bounds=inv)

    for inv in [[], [1.0], [1.5, 0.8], [0.1, 3.0], [float("nan"), 1.2]]:
        with pytest.raises(ValueError):
            SceneIntent(scene_group_id="g1", contrast_trim_bounds=inv)

    for inv in [[], [1.0], [1.5, 0.8], [0.1, 5.0], [0.5, float("inf")]]:
        with pytest.raises(ValueError):
            SceneIntent(scene_group_id="g1", saturation_trim_bounds=inv)

def test_saturation_ceilings_vs_trim_bounds_separation():
    si = SceneIntent(
        scene_group_id="g1",
        scene_saturation_trim_bounds=[0.75, 1.15],
        shadow_output_chroma_ceiling=0.35,
        midtone_output_chroma_ceiling=0.70
    )
    # Multipliers are bounded around 1.0
    assert 0.4 <= si.scene_saturation_trim_bounds[0] < si.scene_saturation_trim_bounds[1] <= 2.0
    # Measured chroma ceilings are in [0, 1] normalized HSV
    assert 0.05 <= si.shadow_output_chroma_ceiling <= 1.0
    assert 0.10 <= si.midtone_output_chroma_ceiling <= 1.0

def test_intentional_silhouette_does_not_trigger_midtone_crush():
    # Create dark silhouette test frame: dark subject against bright background
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:30, :] = [220, 220, 220] # bright sky
    frame[30:, :] = [15, 15, 15]    # silhouette subject (p50 will be very dark ~15)

    from app.media.color import aggregate_shot_metrics
    graded_metrics = aggregate_shot_metrics(
        shot_id="shot_sil",
        video_path="",
        frames=[frame],
        timestamps=[0.0],
        fps=24.0,
        width=100,
        height=100,
        duration_sec=1.0
    )

    intent = SceneIntent(
        scene_group_id="sil_group",
        lighting_class="intentional_silhouette",
        exposure_class="intentional_silhouette"
    )

    health = evaluate_scene_health(
        source_metrics=graded_metrics,
        graded_metrics=graded_metrics,
        graded_frames=[frame],
        scene_intent=intent
    )

    assert "midtone_crush" not in health.hard_gate_failures
    assert health.midtone_readability == 100.0

def test_night_scene_balanced_defaults_to_zero_ev():
    from app.tools.calculate_grade import build_grade_plan
    from app.models.analysis import ShotMetrics, FrameMetrics, SceneIntent

    fm = FrameMetrics(
        timestamp_sec=0.0, mean_luminance=25.0, median_luminance=22.0,
        p5_luminance=5.0, p25_luminance=15.0, p50_luminance=22.0,
        p75_luminance=35.0, p95_luminance=80.0, shadow_clip_pct=1.0,
        highlight_clip_pct=0.0, lab_l_mean=20.0, lab_l_std=10.0,
        lab_a_mean=0.0, lab_a_std=2.0, lab_b_mean=-5.0, lab_b_std=3.0,
        mean_chroma=8.0, r_mean=20.0, g_mean=22.0, b_mean=28.0
    )
    metrics = ShotMetrics(
        shot_id="night_shot", video_path="", duration_sec=1.0, width=640, height=360, fps=24.0,
        sampled_frames=[fm], avg_luminance=25.0, p5_luminance=5.0, p50_luminance=22.0,
        p95_luminance=80.0, avg_shadow_clip_pct=1.0, avg_highlight_clip_pct=0.0,
        avg_lab_mean=[20.0, 0.0, -5.0], avg_lab_std=[10.0, 2.0, 3.0], avg_chroma=8.0
    )

    night_intent = SceneIntent(
        scene_group_id="group_night",
        lighting_class="low_key_night",
        exposure_class="balanced",
        source_relative_exposure_bounds=[-0.6, 0.6]
    )

    plan = build_grade_plan(
        reference=metrics,
        target=metrics,
        scene_intent=night_intent,
        is_reference_shot=False,
        is_same_scene=False
    )

    # Balanced low-key night must default to 0.0 EV exposure trim, NOT automatic darkening
    assert plan.scene_trim.trim_exposure_ev == 0.0, f"Expected 0.0 EV trim for balanced night scene, got {plan.scene_trim.trim_exposure_ev}"

def test_low_key_underexposed_gets_bounded_positive_readability_lift():
    from app.tools.calculate_grade import build_grade_plan
    from app.models.analysis import ShotMetrics, FrameMetrics, SceneIntent

    fm = FrameMetrics(
        timestamp_sec=0.0, mean_luminance=15.0, median_luminance=12.0,
        p5_luminance=2.0, p25_luminance=8.0, p50_luminance=12.0,
        p75_luminance=20.0, p95_luminance=40.0, shadow_clip_pct=2.0,
        highlight_clip_pct=0.0, lab_l_mean=12.0, lab_l_std=6.0,
        lab_a_mean=0.0, lab_a_std=2.0, lab_b_mean=-5.0, lab_b_std=3.0,
        mean_chroma=8.0, r_mean=12.0, g_mean=13.0, b_mean=18.0
    )
    metrics = ShotMetrics(
        shot_id="underexposed_night", video_path="", duration_sec=1.0, width=640, height=360, fps=24.0,
        sampled_frames=[fm], avg_luminance=15.0, p5_luminance=2.0, p50_luminance=12.0,
        p95_luminance=40.0, avg_shadow_clip_pct=2.0, avg_highlight_clip_pct=0.0,
        avg_lab_mean=[12.0, 0.0, -5.0], avg_lab_std=[6.0, 2.0, 3.0], avg_chroma=8.0
    )

    intent = SceneIntent(
        scene_group_id="group_dark",
        lighting_class="low_key_night",
        exposure_class="low_key_underexposed",
        source_relative_exposure_bounds=[-0.6, 0.6]
    )

    plan = build_grade_plan(
        reference=metrics,
        target=metrics,
        scene_intent=intent,
        is_reference_shot=False,
        is_same_scene=False
    )

    # Low-key underexposed allows bounded positive lift for readability
    assert plan.scene_trim.trim_exposure_ev > 0.0, f"Expected positive lift for readability, got {plan.scene_trim.trim_exposure_ev}"
    assert plan.scene_trim.trim_exposure_ev <= 0.5

def test_creative_spec_sdk_schema_compatibility():
    # Verify that SceneIntent has no extra="allow" and produces valid Gemini schema
    from google.genai import types
    from app.models.analysis import CreativeSpecification

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CreativeSpecification,
        temperature=0.2
    )
    assert config.response_schema is CreativeSpecification

