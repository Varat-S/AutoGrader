import pytest
import numpy as np
from app.media.color import (
    sony_slog3_to_linear,
    linear_to_sony_slog3,
    apple_log_to_linear,
    linear_to_apple_log,
    apply_input_camera_profile,
    assess_normalization_health,
    aggregate_shot_metrics,
    MAT_SGAMUT3CINE_TO_BT709,
    MAT_BT2020_TO_BT709
)

def test_sony_slog3_golden_code_values():
    # Test values documented in Sony Technical Summary for S-Log3
    # 0% reflection (black) -> 10-bit code value 95.0 -> normalized 95/1023 ~ 0.092864
    # 18% gray -> 10-bit code value 420.0 -> normalized 420/1023 ~ 0.410557
    # 90% white -> 10-bit code value 597.9 -> normalized ~ 0.58445
    
    linear_test = np.array([0.0, 0.18, 0.90, 1.0, 2.0])
    slog3_encoded = linear_to_sony_slog3(linear_test)
    
    # 1. Check code values
    assert abs(slog3_encoded[0] * 1023.0 - 95.0) < 0.1, "S-Log3 black must be 95 in 10-bit"
    assert abs(slog3_encoded[1] * 1023.0 - 420.0) < 0.1, "S-Log3 18% gray must be 420 in 10-bit"
    assert abs(slog3_encoded[2] * 1023.0 - 597.9) < 0.5, "S-Log3 90% white must be ~598 in 10-bit"
    
    # 2. Check round-trip inversion
    linear_recovered = sony_slog3_to_linear(slog3_encoded)
    np.testing.assert_allclose(linear_test, linear_recovered, atol=1e-5)

def test_sony_sgamut3cine_matrix_properties():
    # S-Gamut3.Cine to BT.709 matrix must preserve neutral white (row sums == 1.0)
    row_sums = np.sum(MAT_SGAMUT3CINE_TO_BT709, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-4)

def test_apple_log_golden_code_values():
    # Test values documented in Apple Log Profile White Paper (2023)
    # 0% black -> encoded ~ 0.15048
    # 18% gray -> encoded ~ 0.48827
    # 100% white -> encoded ~ 0.69455
    
    linear_test = np.array([0.0, 0.01, 0.18, 0.50, 1.0])
    apple_encoded = linear_to_apple_log(linear_test)
    
    assert abs(apple_encoded[0] - 0.15048) < 1e-3, "Apple Log black code value mismatch"
    assert abs(apple_encoded[2] - 0.48827) < 1e-3, "Apple Log 18% gray code value mismatch"
    assert abs(apple_encoded[4] - 0.69455) < 1e-3, "Apple Log 100% white code value mismatch"
    
    # Check round-trip inversion
    linear_recovered = apple_log_to_linear(apple_encoded)
    np.testing.assert_allclose(linear_test, linear_recovered, atol=1e-4)

def test_apple_gamut_matrix_properties():
    # BT.2020 to BT.709 matrix must preserve neutral white (row sums == 1.0)
    row_sums = np.sum(MAT_BT2020_TO_BT709, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-4)

def test_camera_profile_dispatcher():
    # S-Log3 18% gray (norm 0.4105)
    slog_frame = np.full((50, 50, 3), 0.4105, dtype=np.float32)
    out_slog = apply_input_camera_profile(slog_frame, "sony_slog3_sgamut3cine")
    # Must map to standard Rec.709 midtone (~0.40 - 0.42)
    assert 0.38 <= np.mean(out_slog) <= 0.44

    # Rec.709 display ready frame must not be altered
    rec_frame = np.full((50, 50, 3), 0.50, dtype=np.float32)
    out_rec = apply_input_camera_profile(rec_frame, "rec709")
    np.testing.assert_allclose(rec_frame, out_rec, atol=1e-6)

def test_normalization_health_gate():
    # Simulate flat Log frame with elevated blacks (p5 ~ 42)
    flat_frame = np.full((50, 50, 3), 42, dtype=np.uint8)
    metrics = aggregate_shot_metrics("flat", "flat.mp4", [flat_frame], [0.0], 30.0, 50, 50, 1.0)
    
    # Under Rec.709 selection, normalization health must flag PROFILE_CONFIRMATION_REQUIRED
    res = assess_normalization_health("flat", metrics, [flat_frame], profile="rec709")
    assert res.state == "PROFILE_CONFIRMATION_REQUIRED"
    assert not res.passed