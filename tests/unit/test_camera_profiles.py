import pytest
import numpy as np
from app.media.color import (
    sony_slog3_to_linear,
    linear_to_sony_slog3,
    apple_log_to_linear,
    linear_to_apple_log,
    dji_dlog_to_linear,
    linear_to_dji_dlog,
    apply_input_camera_profile,
    assess_normalization_health,
    aggregate_shot_metrics,
    MAT_SGAMUT3CINE_TO_BT709,
    MAT_BT2020_TO_BT709,
    MAT_DGAMUT_TO_BT709
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
    # 1. Neutral white/gray preservation (row sums == 1.0)
    row_sums = np.sum(MAT_SGAMUT3CINE_TO_BT709, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-4)

    # 2. Golden chromatic vectors (Red, Green, Blue primaries & mixed chromatic vector)
    v_red = np.array([1.0, 0.0, 0.0])
    v_green = np.array([0.0, 1.0, 0.0])
    v_blue = np.array([0.0, 0.0, 1.0])
    v_mixed = np.array([0.8, 0.5, 0.2])

    out_red = MAT_SGAMUT3CINE_TO_BT709 @ v_red
    out_green = MAT_SGAMUT3CINE_TO_BT709 @ v_green
    out_blue = MAT_SGAMUT3CINE_TO_BT709 @ v_blue
    out_mixed = MAT_SGAMUT3CINE_TO_BT709 @ v_mixed

    np.testing.assert_allclose(out_red, [1.6586, -0.2100, -0.0195], atol=1e-4)
    np.testing.assert_allclose(out_green, [-0.4939, 1.2583, -0.2521], atol=1e-4)
    np.testing.assert_allclose(out_blue, [-0.1647, -0.0483, 1.2716], atol=1e-4)
    expected_mixed = np.array([
        1.6586 * 0.8 - 0.4939 * 0.5 - 0.1647 * 0.2,
        -0.2100 * 0.8 + 1.2583 * 0.5 - 0.0483 * 0.2,
        -0.0195 * 0.8 - 0.2521 * 0.5 + 1.2716 * 0.2
    ])
    np.testing.assert_allclose(out_mixed, expected_mixed, atol=1e-4)

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
    # 1. BT.2020 to BT.709 matrix neutral white/gray preservation (row sums == 1.0)
    row_sums = np.sum(MAT_BT2020_TO_BT709, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-3)

    # 2. Golden chromatic vectors (Red, Green, Blue primaries & mixed chromatic vector)
    v_red = np.array([1.0, 0.0, 0.0])
    v_green = np.array([0.0, 1.0, 0.0])
    v_blue = np.array([0.0, 0.0, 1.0])
    v_mixed = np.array([0.8, 0.5, 0.2])

    out_red = MAT_BT2020_TO_BT709 @ v_red
    out_green = MAT_BT2020_TO_BT709 @ v_green
    out_blue = MAT_BT2020_TO_BT709 @ v_blue
    out_mixed = MAT_BT2020_TO_BT709 @ v_mixed

    np.testing.assert_allclose(out_red, [1.6605, -0.1246, -0.0182], atol=1e-4)
    np.testing.assert_allclose(out_green, [-0.5876, 1.1329, -0.1006], atol=1e-4)
    np.testing.assert_allclose(out_blue, [-0.0728, -0.0083, 1.1187], atol=1e-4)
    expected_mixed = np.array([
        1.6605 * 0.8 - 0.5876 * 0.5 - 0.0728 * 0.2,
        -0.1246 * 0.8 + 1.1329 * 0.5 - 0.0083 * 0.2,
        -0.0182 * 0.8 - 0.1006 * 0.5 + 1.1187 * 0.2
    ])
    np.testing.assert_allclose(out_mixed, expected_mixed, atol=1e-4)

def test_dji_dlog_golden_code_values():
    # Test values documented in DJI White Paper on D-Log and D-Gamut
    # 0% black (reflection 0.0) -> y = 6.025 * 0 + 0.0929 = 0.0929
    # 18% middle gray (reflection 0.18) -> y = log10(0.18 * 0.9892 + 0.0108) * 0.256663 + 0.584555 ~ 0.39876
    # 100% white (reflection 1.0) -> y = log10(1.0) * 0.256663 + 0.584555 = 0.584555
    linear_test = np.array([0.0, 0.005, 0.0078, 0.01, 0.18, 0.50, 1.0, 5.0, 20.0])
    dlog_encoded = linear_to_dji_dlog(linear_test)

    assert abs(dlog_encoded[0] - 0.0929) < 1e-4, "DJI D-Log black code value mismatch"
    assert abs(dlog_encoded[4] - 0.3988) < 1e-3, "DJI D-Log 18% gray code value mismatch"
    assert abs(dlog_encoded[6] - 0.584555) < 1e-4, "DJI D-Log 100% white code value mismatch"

    # Check round-trip inversion across full dynamic range
    linear_recovered = dji_dlog_to_linear(dlog_encoded)
    np.testing.assert_allclose(linear_test, linear_recovered, atol=1e-4)

def test_dji_dgamut_matrix_properties():
    # 1. D-Gamut to BT.709 matrix neutral white/gray preservation (row sums == 1.0)
    row_sums = np.sum(MAT_DGAMUT_TO_BT709, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-3)

    # 2. Golden chromatic vectors (Red, Green, Blue primaries & mixed chromatic vector)
    v_red = np.array([1.0, 0.0, 0.0])
    v_green = np.array([0.0, 1.0, 0.0])
    v_blue = np.array([0.0, 0.0, 1.0])
    v_mixed = np.array([0.8, 0.5, 0.2])

    out_red = MAT_DGAMUT_TO_BT709 @ v_red
    out_green = MAT_DGAMUT_TO_BT709 @ v_green
    out_blue = MAT_DGAMUT_TO_BT709 @ v_blue
    out_mixed = MAT_DGAMUT_TO_BT709 @ v_mixed

    np.testing.assert_allclose(out_red, [1.6746, -0.0981, -0.0410], atol=1e-4)
    np.testing.assert_allclose(out_green, [-0.5797, 1.3340, -0.2430], atol=1e-4)
    np.testing.assert_allclose(out_blue, [-0.0949, -0.2359, 1.2840], atol=1e-4)
    expected_mixed = np.array([
        1.6746 * 0.8 - 0.5797 * 0.5 - 0.0949 * 0.2,
        -0.0981 * 0.8 + 1.3340 * 0.5 - 0.2359 * 0.2,
        -0.0410 * 0.8 - 0.2430 * 0.5 + 1.2840 * 0.2
    ])
    np.testing.assert_allclose(out_mixed, expected_mixed, atol=1e-4)

def test_camera_profile_dispatcher():
    # S-Log3 18% gray (norm 0.4105)
    slog_frame = np.full((50, 50, 3), 0.4105, dtype=np.float32)
    out_slog = apply_input_camera_profile(slog_frame, "sony_slog3_sgamut3cine")
    # Must map to standard Rec.709 midtone (~0.38 - 0.44)
    assert 0.38 <= np.mean(out_slog) <= 0.44

    # Apple Log / Rec.2020 18% gray (norm 0.4883)
    apple_frame = np.full((50, 50, 3), 0.4883, dtype=np.float32)
    out_apple = apply_input_camera_profile(apple_frame, "apple_log_rec2020")
    assert 0.38 <= np.mean(out_apple) <= 0.46

    # Legacy Apple Log name alias compatibility
    out_apple_alias = apply_input_camera_profile(apple_frame, "apple_log_apple_wide_gamut")
    np.testing.assert_allclose(out_apple, out_apple_alias, atol=1e-6)

    # DJI D-Log / D-Gamut 18% gray (norm ~0.3988)
    dlog_frame = np.full((50, 50, 3), 0.3988, dtype=np.float32)
    out_dlog = apply_input_camera_profile(dlog_frame, "dji_dlog_dgamut")
    assert 0.38 <= np.mean(out_dlog) <= 0.46

    # DJI D-Log aliases
    out_dlog_alias = apply_input_camera_profile(dlog_frame, "dji_dlog")
    np.testing.assert_allclose(out_dlog, out_dlog_alias, atol=1e-6)
    out_dlog_alias2 = apply_input_camera_profile(dlog_frame, "dlog")
    np.testing.assert_allclose(out_dlog, out_dlog_alias2, atol=1e-6)

    # Rec.709 display ready frame must not be altered
    rec_frame = np.full((50, 50, 3), 0.50, dtype=np.float32)
    out_rec = apply_input_camera_profile(rec_frame, "rec709")
    np.testing.assert_allclose(rec_frame, out_rec, atol=1e-6)

    # auto_ask must be rejected with ValueError
    with pytest.raises(ValueError, match="auto_ask is a pending decision state"):
        apply_input_camera_profile(slog_frame, "auto_ask")

    # Unknown profile string must be rejected with ValueError
    with pytest.raises(ValueError, match="Unknown camera input profile"):
        apply_input_camera_profile(slog_frame, "unknown_custom_log")

def test_normalization_health_gate():
    # Simulate flat Log frame with elevated blacks (p5 ~ 42)
    flat_frame = np.full((50, 50, 3), 42, dtype=np.uint8)
    metrics = aggregate_shot_metrics("flat", "flat.mp4", [flat_frame], [0.0], 30.0, 50, 50, 1.0)
    
    # Under non-blocking normalization, health checks return advisory warning and pass to guarantee delivery
    res = assess_normalization_health("flat", metrics, [flat_frame], profile="rec709")
    assert res.state == "NORMALIZATION_WARNING"
    assert res.passed is True

def test_dji_dlog_m_golden_code_values_and_reversibility():
    from app.media.color import dji_dlog_m_to_linear, linear_to_dji_dlog_m
    linear_test = np.array([0.0, 0.01, 0.05, 0.18, 0.50, 1.0, 2.0, 5.0], dtype=np.float32)
    dlog_m_encoded = linear_to_dji_dlog_m(linear_test)

    # Check black floor is 0.10
    assert abs(dlog_m_encoded[0] - 0.10) < 1e-4, "D-Log M black floor must be 0.10"
    # Check 18% middle gray is near 0.46
    assert 0.44 <= dlog_m_encoded[3] <= 0.48, "D-Log M 18% gray should be ~0.46"
    # Check 50% luminance is near 0.72
    assert 0.70 <= dlog_m_encoded[4] <= 0.75, "D-Log M 50% luminance should be ~0.72"
    # Check 100% white is 1.0
    assert abs(dlog_m_encoded[5] - 1.0) < 1e-4, "D-Log M 100% white should be 1.0"

    # Reversibility test
    linear_recovered = dji_dlog_m_to_linear(dlog_m_encoded)
    np.testing.assert_allclose(linear_test, linear_recovered, atol=1e-5)

def test_dji_dlog_m_dispatcher():
    # D-Log M 18% gray (norm ~0.46)
    dlog_m_frame = np.full((30, 30, 3), 0.46, dtype=np.float32)
    out_dlog_m = apply_input_camera_profile(dlog_m_frame, "dji_dlog_m_rec709")
    # Must map to standard Rec.709 midtone (~0.38 - 0.46)
    assert 0.38 <= np.mean(out_dlog_m) <= 0.46

    # Aliases
    out_alias = apply_input_camera_profile(dlog_m_frame, "dji_dlog_m")
    np.testing.assert_allclose(out_dlog_m, out_alias, atol=1e-6)
    out_alias2 = apply_input_camera_profile(dlog_m_frame, "dlog_m")
    np.testing.assert_allclose(out_dlog_m, out_alias2, atol=1e-6)

def test_luminance_preserving_gamut_compression():
    from app.media.color import compress_out_of_gamut_rgb, REC709_LUMA_COEFFS
    # Construct an out-of-gamut pixel where blue channel is negative (e.g. wide gamut saturated red/green)
    rgb = np.array([[[1.2, 0.4, -0.3]]], dtype=np.float32)
    orig_y = np.sum(rgb * REC709_LUMA_COEFFS, axis=-1, keepdims=True)

    compressed, neg_pct, mean_comp = compress_out_of_gamut_rgb(rgb)
    new_y = np.sum(compressed * REC709_LUMA_COEFFS, axis=-1, keepdims=True)

    # 1. Luminance must be 100% preserved
    np.testing.assert_allclose(orig_y, new_y, atol=1e-6)
    # 2. Minimum channel must be brought to >= 0
    assert np.min(compressed) >= -1e-6
    assert neg_pct > 0.0
    # 3. Already in-gamut pixels must be unaltered
    in_gamut = np.array([[[0.5, 0.6, 0.7]]], dtype=np.float32)
    compressed_in, neg_pct_in, _ = compress_out_of_gamut_rgb(in_gamut)
    np.testing.assert_allclose(in_gamut, compressed_in, atol=1e-6)
    assert neg_pct_in == 0.0