import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.analysis import ShotMetrics
from app.models.grade import (
    InputProfile,
    ShotProfileSelection,
    GradePlan,
    InputTransformParams
)
from app.tools.calculate_grade import assess_input_profile, build_grade_plan
from app.media.color import apply_input_camera_profile

client = TestClient(app)

def make_metrics(
    shot_id="shot_A",
    p5=45.0,
    p25=60.0,
    p75=120.0,
    p95=180.0,
    chroma=8.5,
    avg_lum=100.0
):
    return ShotMetrics(
        shot_id=shot_id,
        video_path="test.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        duration_sec=3.0,
        sampled_frames=[],
        avg_luminance=avg_lum,
        median_luminance=avg_lum,
        p5_luminance=p5,
        p25_luminance=p25,
        p75_luminance=p75,
        p95_luminance=p95,
        avg_chroma=chroma,
        avg_shadow_clip_pct=0.0,
        avg_highlight_clip_pct=0.0,
        avg_lab_mean=[50.0, 0.0, 0.0],
        avg_lab_std=[15.0, 5.0, 5.0],
        dominant_cast="neutral"
    )

def test_slog3_metadata_auto_ask():
    # S-Log3 metadata + auto_ask -> recommends Sony S-Log3, requires confirmation, unresolved
    probed = {"color_transfer": "s-log3", "color_primaries": "s-gamut3.cine"}
    metrics = make_metrics()
    decision = assess_input_profile("shot_A", probed, metrics, requested_profile="auto_ask", shot_index=0)
    assert decision.recommended_profile == "sony_slog3_sgamut3cine"
    assert decision.metadata_recommendation == "sony_slog3_sgamut3cine"
    assert decision.requires_confirmation is True
    assert decision.resolved_profile is None
    assert decision.resolution_source == "unresolved"

def test_apple_log_metadata_auto_ask():
    # Apple Log metadata + auto_ask -> recommends Apple Log / Rec.2020
    probed = {"color_transfer": "arib-std-b67", "color_primaries": "bt2020"}
    metrics = make_metrics(p5=40.0, p25=55.0, p75=115.0, p95=175.0, chroma=9.0)
    decision = assess_input_profile("shot_A", probed, metrics, requested_profile="auto_ask", shot_index=0)
    assert decision.recommended_profile == "apple_log_rec2020"
    assert decision.metadata_recommendation == "apple_log_rec2020"
    assert decision.requires_confirmation is True
    assert decision.resolved_profile is None

def test_histogram_only_conservative_detector():
    # Elevated p5, low IQR, low chroma with NO metadata
    # Must NEVER definitively declare Sony S-Log3 or Apple Log from pixel values alone!
    probed = {}
    metrics = make_metrics(p5=42.0, p25=55.0, p75=95.0, p95=160.0, chroma=9.0, avg_lum=90.0)
    decision = assess_input_profile("shot_A", probed, metrics, requested_profile="auto_ask", shot_index=0)
    assert decision.signal_class_hint == "log_like"
    assert decision.recommended_profile == "generic_log_experimental"
    assert decision.metadata_recommendation is None
    assert decision.requires_confirmation is True
    assert decision.resolved_profile is None
    # Must NOT guess specific camera hardware
    assert decision.recommended_profile not in ["sony_slog3_sgamut3cine", "apple_log_rec2020"]

def test_user_confirmed_profile_retention():
    probed = {"color_transfer": "s-log3"}
    metrics = make_metrics()
    # 1. User confirmed recommendation
    decision = assess_input_profile("shot_A", probed, metrics, requested_profile="sony_slog3_sgamut3cine", user_confirmed=True)
    assert decision.resolved_profile == "sony_slog3_sgamut3cine"
    assert decision.requires_confirmation is False
    assert decision.resolution_source == "user_explicit"

    # 2. User explicitly confirmed auto_ask recommendation
    decision_conf = assess_input_profile("shot_A", probed, metrics, requested_profile="auto_ask", user_confirmed=True)
    assert decision_conf.resolved_profile == "sony_slog3_sgamut3cine"
    assert decision_conf.requires_confirmation is False
    assert decision_conf.resolution_source == "user_confirmed"

def test_auto_ask_rejected_from_grade_plan():
    # auto_ask must raise ValueError when constructing GradePlan
    metrics = make_metrics()
    with pytest.raises(ValueError, match="auto_ask is a pending decision state"):
        build_grade_plan(metrics, metrics, color_profile="auto_ask")

    with pytest.raises(ValueError, match="auto_ask is a pending decision state"):
        InputTransformParams(profile="auto_ask")

def test_api_rejection_of_unresolved_auto_ask():
    # Create job and load demo
    res = client.post("/api/jobs")
    job_id = res.json()["job_id"]
    client.post(f"/api/jobs/{job_id}/load_demo")

    # Run with auto_ask profile
    res_run = client.post(
        f"/api/jobs/{job_id}/run",
        json={
            "creative_prompt": "cinematic look",
            "reference_index": 0,
            "color_profile": "rec709",
            "input_profiles": [
                {"shot_index": 0, "profile": "auto_ask"},
                {"shot_index": 1, "profile": "rec709"},
                {"shot_index": 2, "profile": "rec709"}
            ]
        }
    )
    assert res_run.status_code == 409
    assert "confirmation required" in res_run.json()["detail"].lower()

def test_api_rejection_of_invalid_and_duplicate_indices():
    res = client.post("/api/jobs")
    job_id = res.json()["job_id"]
    client.post(f"/api/jobs/{job_id}/load_demo")

    # 1. Out of range index
    res_oor = client.post(
        f"/api/jobs/{job_id}/run",
        json={
            "creative_prompt": "cinematic look",
            "reference_index": 0,
            "color_profile": "rec709",
            "input_profiles": [
                {"shot_index": 99, "profile": "rec709"}
            ]
        }
    )
    assert res_oor.status_code == 400
    assert "out of range" in res_oor.json()["detail"]

    # 2. Duplicate shot_index
    res_dup = client.post(
        f"/api/jobs/{job_id}/run",
        json={
            "creative_prompt": "cinematic look",
            "reference_index": 0,
            "color_profile": "rec709",
            "input_profiles": [
                {"shot_index": 0, "profile": "rec709"},
                {"shot_index": 0, "profile": "rec709"}
            ]
        }
    )
    assert res_dup.status_code == 400
    assert "Duplicate" in res_dup.json()["detail"]

    # 3. Invalid profile name rejected with 422
    res_inv = client.post(
        f"/api/jobs/{job_id}/run",
        json={
            "creative_prompt": "cinematic look",
            "reference_index": 0,
            "color_profile": "rec709",
            "input_profiles": [
                {"shot_index": 0, "profile": "dlog_custom"}
            ]
        }
    )
    assert res_inv.status_code == 422