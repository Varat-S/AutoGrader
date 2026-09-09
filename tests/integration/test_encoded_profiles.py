import os
import cv2
import numpy as np
import pytest
from unittest.mock import patch

from app.agent import AutonomousColoristAgent
from app.models.grade import InputProfile, ShotProfileSelection
from app.models.analysis import (
    SequenceInspectionResult,
    ShotSemanticAnalysis,
    CinematographyResearchResult,
    CreativeSpecification,
    SearchCitation,
)
from app.media.color import (
    linear_to_sony_slog3,
    linear_to_apple_log,
    linear_to_dji_dlog,
    apply_input_camera_profile,
    assess_normalization_health,
    aggregate_shot_metrics,
    MAT_SGAMUT3CINE_TO_BT709,
    MAT_BT2020_TO_BT709,
    MAT_DGAMUT_TO_BT709,
)

MAT_BT709_TO_SGAMUT3CINE = np.linalg.inv(MAT_SGAMUT3CINE_TO_BT709)
MAT_BT709_TO_BT2020 = np.linalg.inv(MAT_BT2020_TO_BT709)
MAT_BT709_TO_DGAMUT = np.linalg.inv(MAT_DGAMUT_TO_BT709)

def generate_synthetic_scene_linear_gradient(height=64, width=64) -> np.ndarray:
    """Creates a scene-linear RGB image spanning deep shadow to highlights."""
    gradient = np.linspace(0.005, 1.2, width, dtype=np.float32)
    img_linear = np.tile(gradient, (height, 1))
    rgb_linear = np.stack([img_linear, img_linear * 0.9, img_linear * 0.8], axis=-1)
    return rgb_linear

def generate_slog3_frame(height=64, width=64) -> np.ndarray:
    """Synthesizes an 8-bit BGR frame encoded in S-Log3 / S-Gamut3.Cine."""
    rgb_linear = generate_synthetic_scene_linear_gradient(height, width)
    h, w, c = rgb_linear.shape
    sgamut = np.dot(rgb_linear.reshape(-1, 3), MAT_BT709_TO_SGAMUT3CINE.T).reshape(h, w, c)
    sgamut = np.maximum(0.0, sgamut)
    slog3 = linear_to_sony_slog3(sgamut)
    slog3_bgr = cv2.cvtColor(np.clip(slog3, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2BGR)
    return (np.clip(slog3_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)

def generate_apple_log_frame(height=64, width=64) -> np.ndarray:
    """Synthesizes an 8-bit BGR frame encoded in Apple Log / Rec.2020."""
    rgb_linear = generate_synthetic_scene_linear_gradient(height, width)
    h, w, c = rgb_linear.shape
    bt2020 = np.dot(rgb_linear.reshape(-1, 3), MAT_BT709_TO_BT2020.T).reshape(h, w, c)
    bt2020 = np.maximum(0.0, bt2020)
    apple_log = linear_to_apple_log(bt2020)
    apple_log_bgr = cv2.cvtColor(np.clip(apple_log, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2BGR)
    return (np.clip(apple_log_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)

def generate_dji_dlog_frame(height=64, width=64) -> np.ndarray:
    """Synthesizes an 8-bit BGR frame encoded in DJI D-Log / D-Gamut."""
    rgb_linear = generate_synthetic_scene_linear_gradient(height, width)
    h, w, c = rgb_linear.shape
    dgamut = np.dot(rgb_linear.reshape(-1, 3), MAT_BT709_TO_DGAMUT.T).reshape(h, w, c)
    dgamut = np.maximum(0.0, dgamut)
    dlog = linear_to_dji_dlog(dgamut)
    dlog_bgr = cv2.cvtColor(np.clip(dlog, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2BGR)
    return (np.clip(dlog_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)

def create_synthetic_video(filepath: str, frame_generator, num_frames=12, fps=24.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (64, 64))
    for _ in range(num_frames):
        frame = frame_generator(64, 64)
        writer.write(frame)
    writer.release()

def test_slog3_tonal_expansion_and_recovery():
    frame_slog3 = generate_slog3_frame()
    gray_raw = cv2.cvtColor(frame_slog3, cv2.COLOR_BGR2GRAY)
    p5_raw = float(np.percentile(gray_raw, 5))
    p75_raw = float(np.percentile(gray_raw, 75))
    p25_raw = float(np.percentile(gray_raw, 25))
    iqr_raw = p75_raw - p25_raw
    assert p5_raw >= 20.0, f"S-Log3 raw black floor should be elevated, got {p5_raw}"

    frame_float = frame_slog3.astype(np.float32) / 255.0
    normalized = apply_input_camera_profile(frame_float, profile="sony_slog3")
    norm_uint8 = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray_norm = cv2.cvtColor(norm_uint8, cv2.COLOR_BGR2GRAY)

    p5_norm = float(np.percentile(gray_norm, 5))
    p75_norm = float(np.percentile(gray_norm, 75))
    p25_norm = float(np.percentile(gray_norm, 25))
    iqr_norm = p75_norm - p25_norm

    assert p5_norm < p5_raw, f"Normalized p5 ({p5_norm}) should be lower than raw S-Log3 ({p5_raw})"
    assert iqr_norm > iqr_raw * 1.2, f"IQR should expand after S-Log3 normalization (raw: {iqr_raw}, norm: {iqr_norm})"

def test_apple_log_tonal_expansion_and_recovery():
    frame_apple = generate_apple_log_frame()
    gray_raw = cv2.cvtColor(frame_apple, cv2.COLOR_BGR2GRAY)
    p5_raw = float(np.percentile(gray_raw, 5))
    p75_raw = float(np.percentile(gray_raw, 75))
    p25_raw = float(np.percentile(gray_raw, 25))
    iqr_raw = p75_raw - p25_raw
    assert p5_raw >= 18.0, f"Apple Log raw black floor should be elevated, got {p5_raw}"

    frame_float = frame_apple.astype(np.float32) / 255.0
    normalized = apply_input_camera_profile(frame_float, profile="apple_log_rec2020")
    norm_uint8 = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray_norm = cv2.cvtColor(norm_uint8, cv2.COLOR_BGR2GRAY)

    p5_norm = float(np.percentile(gray_norm, 5))
    p75_norm = float(np.percentile(gray_norm, 75))
    p25_norm = float(np.percentile(gray_norm, 25))
    iqr_norm = p75_norm - p25_norm

    assert p5_norm < p5_raw, f"Normalized p5 ({p5_norm}) should be lower than raw Apple Log ({p5_raw})"
    assert iqr_norm > iqr_raw * 1.15, f"IQR should expand after Apple Log normalization (raw: {iqr_raw}, norm: {iqr_norm})"

def test_dji_dlog_tonal_expansion_and_recovery():
    frame_dlog = generate_dji_dlog_frame()
    gray_raw = cv2.cvtColor(frame_dlog, cv2.COLOR_BGR2GRAY)
    p5_raw = float(np.percentile(gray_raw, 5))
    p75_raw = float(np.percentile(gray_raw, 75))
    p25_raw = float(np.percentile(gray_raw, 25))
    iqr_raw = p75_raw - p25_raw
    assert p5_raw >= 18.0, f"DJI D-Log raw black floor should be elevated, got {p5_raw}"

    frame_float = frame_dlog.astype(np.float32) / 255.0
    normalized = apply_input_camera_profile(frame_float, profile="dji_dlog_dgamut")
    norm_uint8 = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray_norm = cv2.cvtColor(norm_uint8, cv2.COLOR_BGR2GRAY)

    p5_norm = float(np.percentile(gray_norm, 5))
    p75_norm = float(np.percentile(gray_norm, 75))
    p25_norm = float(np.percentile(gray_norm, 25))
    iqr_norm = p75_norm - p25_norm

    assert p5_norm < p5_raw, f"Normalized p5 ({p5_norm}) should be lower than raw DJI D-Log ({p5_raw})"
    assert iqr_norm > iqr_raw * 1.15, f"IQR should expand after DJI D-Log normalization (raw: {iqr_raw}, norm: {iqr_norm})"

def test_agent_sequence_pipeline_with_encoded_log_clips(tmp_path):
    slog3_video = str(tmp_path / "synthetic_slog3.mp4")
    apple_video = str(tmp_path / "synthetic_apple.mp4")
    create_synthetic_video(slog3_video, generate_slog3_frame)
    create_synthetic_video(apple_video, generate_apple_log_frame)

    agent = AutonomousColoristAgent(work_dir=str(tmp_path / "work"))

    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="S-Log3 master shot",
                lighting_environment="controlled studio",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=1.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.92,
            ),
            ShotSemanticAnalysis(
                shot_id="shot_B",
                scene_group_id="group_1",
                relationship_to_reference="same_scene",
                scene_description="Apple Log second angle",
                lighting_environment="controlled studio",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=1.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.85,
            ),
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="same_scene",
    )

    mock_research = CinematographyResearchResult(
        query="studio commercial",
        objective="research",
        sources=[SearchCitation(title="ASC", url="https://theasc.com/article", excerpt="Clean skin tones, natural contrast.")],
        is_grounded=True,
    )

    mock_spec = CreativeSpecification(
        look_title="Clean Commercial",
        target_aesthetic="Natural skin tones, subtle contrast, balanced neutral highlights",
        contrast_intent=1.05,
        saturation_intent=1.02,
        highlight_bias="neutral",
        shadow_bias="neutral",
        black_level_treatment="clean anchor",
        temperature_shift=0.0,
        tint_shift=0.0,
        black_mist_diffusion_strength=0.0,
        cinematography_principles=["Natural contrast", "Clean color balance"],
        citations=mock_research.sources,
    )

    input_profiles = [
        ShotProfileSelection(shot_index=0, profile=InputProfile.SONY_SLOG3, user_confirmed=True),
        ShotProfileSelection(shot_index=1, profile=InputProfile.APPLE_LOG, user_confirmed=True),
    ]

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=[slog3_video, apple_video],
            creative_prompt="Clean studio commercial grade.",
            input_profiles=input_profiles,
            job_id="test_log_encoded_pipeline",
        )

    assert result["job_id"] == "test_log_encoded_pipeline"
    assert len(result["results"]) == 2
    assert len(result["normalization_results"]) == 2
    for n in result["normalization_results"]:
        assert n["passed"] is True
        assert n["state"] == "NORMALIZATION_VERIFIED"
    for r in result["results"]:
        assert os.path.exists(r["output_video_path"])
        assert os.path.exists(r["lut_path"])
