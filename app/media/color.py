import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
from app.models.analysis import (
    FrameMetrics,
    ShotMetrics,
    NormalizationValidationResult,
    NormalizationDiagnostics,
    LookContinuityScore,
    SceneHealthScore,
    SceneIntent
)
from app.models.grade import (
    ColorGradeParams,
    GradePlan,
    ConsistencyScore,
    InputTransformParams,
    TechnicalBalanceParams,
    SceneMatchParams,
    CreativeLookParams,
    SceneTrimParams,
    OutputTransformParams
)

def compute_frame_metrics(bgr_frame: np.ndarray, timestamp_sec: float = 0.0) -> FrameMetrics:
    if bgr_frame.dtype != np.uint8:
        bgr_uint8 = np.clip(bgr_frame * 255.0, 0, 255).astype(np.uint8)
    else:
        bgr_uint8 = bgr_frame
        
    h, w, _ = bgr_uint8.shape
    total_pixels = max(1, h * w)
    
    b, g, r = bgr_uint8[:, :, 0], bgr_uint8[:, :, 1], bgr_uint8[:, :, 2]
    r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    
    # Rec.709 Luminance
    y = 0.2126 * r.astype(np.float32) + 0.7152 * g.astype(np.float32) + 0.0722 * b.astype(np.float32)
    mean_lum = float(np.mean(y))
    median_lum = float(np.median(y))
    p5_lum = float(np.percentile(y, 5))
    p25_lum = float(np.percentile(y, 25))
    p50_lum = float(np.percentile(y, 50))
    p75_lum = float(np.percentile(y, 75))
    p95_lum = float(np.percentile(y, 95))
    
    shadow_clip = float(np.sum(y < 2.0) / total_pixels * 100.0)
    highlight_clip = float(np.sum(y > 253.0) / total_pixels * 100.0)
    
    # Perceptual CIELAB in standard ranges: L in [0, 100], a in [-127, 127], b in [-127, 127]
    lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_std = lab[:, :, 0] * (100.0 / 255.0)
    a_std = lab[:, :, 1] - 128.0
    b_std = lab[:, :, 2] - 128.0
    
    l_mean, l_dev = float(np.mean(l_std)), float(np.std(l_std))
    a_mean, a_dev = float(np.mean(a_std)), float(np.std(a_std))
    b_mean, b_dev = float(np.mean(b_std)), float(np.std(b_std))
    
    chroma = np.sqrt(a_std**2 + b_std**2)
    mean_chroma = float(np.mean(chroma))
    
    return FrameMetrics(
        timestamp_sec=timestamp_sec,
        mean_luminance=round(mean_lum, 2),
        median_luminance=round(median_lum, 2),
        p5_luminance=round(p5_lum, 2),
        p25_luminance=round(p25_lum, 2),
        p50_luminance=round(p50_lum, 2),
        p75_luminance=round(p75_lum, 2),
        p95_luminance=round(p95_lum, 2),
        shadow_clip_pct=round(shadow_clip, 2),
        highlight_clip_pct=round(highlight_clip, 2),
        lab_l_mean=round(l_mean, 2),
        lab_l_std=round(l_dev, 2),
        lab_a_mean=round(a_mean, 2),
        lab_a_std=round(a_dev, 2),
        lab_b_mean=round(b_mean, 2),
        lab_b_std=round(b_dev, 2),
        mean_chroma=round(mean_chroma, 2),
        r_mean=round(r_mean, 2),
        g_mean=round(g_mean, 2),
        b_mean=round(b_mean, 2)
    )

def aggregate_shot_metrics(
    shot_id: str,
    video_path: str,
    frames: List[np.ndarray],
    timestamps: List[float],
    fps: float,
    width: int,
    height: int,
    duration_sec: float
) -> ShotMetrics:
    frame_metrics = [compute_frame_metrics(f, t) for f, t in zip(frames, timestamps)]
    avg_lum = float(np.mean([m.mean_luminance for m in frame_metrics])) if frame_metrics else 0.0
    p5_avg = float(np.mean([m.p5_luminance for m in frame_metrics])) if frame_metrics else 0.0
    p25_avg = float(np.mean([m.p25_luminance for m in frame_metrics])) if frame_metrics else 0.0
    p50_avg = float(np.mean([m.p50_luminance for m in frame_metrics])) if frame_metrics else 0.0
    p75_avg = float(np.mean([m.p75_luminance for m in frame_metrics])) if frame_metrics else 0.0
    p95_avg = float(np.mean([m.p95_luminance for m in frame_metrics])) if frame_metrics else 0.0
    
    shadow_clip_avg = float(np.mean([m.shadow_clip_pct for m in frame_metrics])) if frame_metrics else 0.0
    highlight_clip_avg = float(np.mean([m.highlight_clip_pct for m in frame_metrics])) if frame_metrics else 0.0
    
    avg_l = float(np.mean([m.lab_l_mean for m in frame_metrics])) if frame_metrics else 0.0
    avg_a = float(np.mean([m.lab_a_mean for m in frame_metrics])) if frame_metrics else 0.0
    avg_b = float(np.mean([m.lab_b_mean for m in frame_metrics])) if frame_metrics else 0.0
    
    avg_l_std = float(np.mean([m.lab_l_std for m in frame_metrics])) if frame_metrics else 0.0
    avg_a_std = float(np.mean([m.lab_a_std for m in frame_metrics])) if frame_metrics else 0.0
    avg_b_std = float(np.mean([m.lab_b_std for m in frame_metrics])) if frame_metrics else 0.0
    avg_chroma = float(np.mean([m.mean_chroma for m in frame_metrics])) if frame_metrics else 0.0
    
    cast = "neutral"
    if avg_b > 10.0 and avg_a > 2.0:
        cast = "warm / golden"
    elif avg_b < -10.0:
        cast = "cool / blue"
    elif avg_a < -8.0:
        cast = "green tint"
    elif avg_a > 10.0:
        cast = "magenta tint"
        
    return ShotMetrics(
        shot_id=shot_id,
        video_path=video_path,
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        sampled_frames=frame_metrics,
        avg_luminance=round(avg_lum, 2),
        p5_luminance=round(p5_avg, 2),
        p25_luminance=round(p25_avg, 2),
        p50_luminance=round(p50_avg, 2),
        p75_luminance=round(p75_avg, 2),
        p95_luminance=round(p95_avg, 2),
        avg_shadow_clip_pct=round(shadow_clip_avg, 2),
        avg_highlight_clip_pct=round(highlight_clip_avg, 2),
        avg_lab_mean=[round(avg_l, 2), round(avg_a, 2), round(avg_b, 2)],
        avg_lab_std=[round(avg_l_std, 2), round(avg_a_std, 2), round(avg_b_std, 2)],
        avg_chroma=round(avg_chroma, 2),
        dominant_cast=cast
    )

def is_log_profile(metrics_or_profile: Union[ShotMetrics, str]) -> bool:
    if isinstance(metrics_or_profile, str):
        p = metrics_or_profile.lower().strip()
        return p not in ["rec709", "rec.709", "bt709", "srgb", "display", "auto"]
    metrics = metrics_or_profile
    p5_val = metrics.p5_luminance if metrics.p5_luminance > 0 else float(np.mean([f.p5_luminance for f in metrics.sampled_frames])) if metrics.sampled_frames else 0.0
    # Must have elevated black floor (>38) AND low baseline chroma (<12) AND flat tonal spread
    iqr = metrics.p75_luminance - metrics.p25_luminance
    return (p5_val > 38.0 and metrics.avg_chroma < 12.0 and iqr < 55.0)

# --- AUTHORITATIVE CAMERA COLOR PROFILES ---

# 1. Sony S-Log3 / S-Gamut3.Cine (Sony Technical Summary for S-Log3)
MAT_SGAMUT3CINE_TO_BT709 = np.array([
    [ 1.6586, -0.4939, -0.1647],
    [-0.2100,  1.2583, -0.0483],
    [-0.0195, -0.2521,  1.2716]
], dtype=np.float32)

def sony_slog3_to_linear(y: np.ndarray) -> np.ndarray:
    """Decodes normalized Sony S-Log3 [0, 1] to scene-linear light."""
    y_cut = 105.70064338 / 1023.0
    return np.where(
        y >= y_cut,
        (10.0 ** ((y * 1023.0 - 420.0) / 261.5)) * 0.19 - 0.01,
        ((y * 1023.0 - 95.0) * 0.18) / 171.21029408
    )

def linear_to_sony_slog3(x: np.ndarray) -> np.ndarray:
    """Encodes scene-linear light to normalized Sony S-Log3 [0, 1]."""
    return np.where(
        x >= 0.01125,
        (420.0 + np.log10(np.maximum(1e-7, (x + 0.01) / 0.19)) * 261.5) / 1023.0,
        ((x * 171.21029408) / 0.18 + 95.0) / 1023.0
    )

# 2. Apple Log / Rec.2020 (Apple Log Profile White Paper 2023, ITU-R BT.2020 gamut to BT.709 gamut)
MAT_BT2020_TO_BT709 = np.array([
    [ 1.6605, -0.5876, -0.0728],
    [-0.1246,  1.1329, -0.0083],
    [-0.0182, -0.1006,  1.1187]
], dtype=np.float32)

APPLE_LOG_R0 = -0.05641088
APPLE_LOG_RT = 0.01
APPLE_LOG_C = 47.28711236
APPLE_LOG_BETA = 0.00964052
APPLE_LOG_GAMMA = 0.08550479
APPLE_LOG_DELTA = 0.69336945
APPLE_LOG_PT = APPLE_LOG_C * ((APPLE_LOG_RT - APPLE_LOG_R0) ** 2)

def apple_log_to_linear(x: np.ndarray) -> np.ndarray:
    """Decodes normalized Apple Log [0, 1] to scene-linear light."""
    return np.where(
        x >= APPLE_LOG_PT,
        (2.0 ** ((x - APPLE_LOG_DELTA) / APPLE_LOG_GAMMA)) - APPLE_LOG_BETA,
        np.where(
            x >= 0.0,
            np.sqrt(np.maximum(0.0, x / APPLE_LOG_C)) + APPLE_LOG_R0,
            APPLE_LOG_R0
        )
    )

def linear_to_apple_log(r: np.ndarray) -> np.ndarray:
    """Encodes scene-linear light to normalized Apple Log [0, 1]."""
    return np.where(
        r >= APPLE_LOG_RT,
        APPLE_LOG_GAMMA * np.log2(np.maximum(1e-9, r + APPLE_LOG_BETA)) + APPLE_LOG_DELTA,
        APPLE_LOG_C * ((np.maximum(APPLE_LOG_R0, r) - APPLE_LOG_R0) ** 2)
    )

# 3. DJI D-Log / D-Gamut (DJI White Paper on D-Log and D-Gamut)
MAT_DGAMUT_TO_BT709 = np.array([
    [ 1.6746, -0.5797, -0.0949],
    [-0.0981,  1.3340, -0.2359],
    [-0.0410, -0.2430,  1.2840]
], dtype=np.float32)

MAT_BT709_TO_DGAMUT = np.linalg.inv(MAT_DGAMUT_TO_BT709).astype(np.float32)

def dji_dlog_to_linear(y: np.ndarray) -> np.ndarray:
    """Decodes normalized DJI D-Log [0, 1] to scene-linear light.
    DJI D-Log curve specification:
    For y <= 0.14: x = (y - 0.0929) / 6.025
    For y >  0.14: x = (10 ** ((y - 0.584555) / 0.256663) - 0.0108) / 0.9892
    """
    return np.where(
        y <= 0.14,
        (y - 0.0929) / 6.025,
        (10.0 ** ((y - 0.584555) / 0.256663) - 0.0108) / 0.9892
    )

def linear_to_dji_dlog(x: np.ndarray) -> np.ndarray:
    """Encodes scene-linear light to normalized DJI D-Log [0, 1].
    DJI D-Log curve specification:
    For x <= 0.0078: y = 6.025 * x + 0.0929
    For x >  0.0078: y = log10(x * 0.9892 + 0.0108) * 0.256663 + 0.584555
    """
    return np.where(
        x <= 0.0078,
        6.025 * x + 0.0929,
        np.log10(np.maximum(1e-9, x * 0.9892 + 0.0108)) * 0.256663 + 0.584555
    )

# 4. DJI D-Log M / Rec.709 (Mavic 3 / Air 2S / Air 3 / Mini 4 Pro / Pocket 3 / Action 4/5)
def dji_dlog_m_to_linear(y: np.ndarray) -> np.ndarray:
    """Decodes normalized DJI D-Log M [0, 1] to scene-linear light.
    DJI D-Log M curve specification:
    Black floor at 0.10 (102/1023), 18% middle gray at ~0.46.
    For y <= 0.10: (y - 0.10) / 4.5
    For y >  0.10: ((y - 0.10) / 0.90) ** 1.90
    """
    return np.where(
        y <= 0.10,
        (y - 0.10) / 4.5,
        np.maximum(0.0, (y - 0.10) / 0.90) ** 1.90
    )

def linear_to_dji_dlog_m(x: np.ndarray) -> np.ndarray:
    """Encodes scene-linear light to normalized DJI D-Log M [0, 1].
    For x <= 0: 0.10 + 4.5 * x
    For x >  0: 0.10 + 0.90 * (x ** (1.0 / 1.90))
    """
    return np.where(
        x <= 0.0,
        0.10 + 4.5 * x,
        0.10 + 0.90 * (np.maximum(0.0, x) ** (1.0 / 1.90))
    )

REC709_LUMA_COEFFS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

def compress_out_of_gamut_rgb(rgb_linear: np.ndarray, eps: float = 1e-7) -> Tuple[np.ndarray, float, float]:
    """Compresses negative (out-of-gamut) channels toward neutral gray while strictly preserving luminance Y.
    
    Returns:
        (compressed_rgb, negative_excursion_pct, mean_compression_amount)
    """
    y_lin = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    min_c = np.min(rgb_linear, axis=-1)
    neg_mask = min_c < 0.0
    neg_pct = float(np.mean(neg_mask) * 100.0)
    
    if not np.any(neg_mask):
        return rgb_linear, 0.0, 0.0
        
    compressed = rgb_linear.copy()
    valid_y_mask = neg_mask & (y_lin > eps)
    if np.any(valid_y_mask):
        y_val = y_lin[valid_y_mask]
        min_val = min_c[valid_y_mask]
        s = np.clip(y_val / (y_val - min_val + eps), 0.0, 1.0)[..., np.newaxis]
        y_expanded = y_val[..., np.newaxis]
        compressed[valid_y_mask] = y_expanded + s * (compressed[valid_y_mask] - y_expanded)
        mean_comp = float(np.mean(1.0 - s))
    else:
        mean_comp = 0.0
        
    nonpos_mask = neg_mask & (y_lin <= eps)
    if np.any(nonpos_mask):
        compressed[nonpos_mask] = np.maximum(0.0, compressed[nonpos_mask])
        
    return compressed, neg_pct, mean_comp

def scene_linear_to_rec709_display(linear_rgb: np.ndarray) -> np.ndarray:
    """Standard ITU-R BT.709 display tone curve with highlight roll-off."""
    threshold = 0.85
    comp = np.where(
        linear_rgb > threshold,
        threshold + (linear_rgb - threshold) / (1.0 + (linear_rgb - threshold) * 0.75),
        linear_rgb
    )
    oetf = np.where(
        comp < 0.018,
        4.5 * comp,
        1.099 * (np.maximum(0.0, comp) ** 0.45) - 0.099
    )
    return np.clip(oetf, 0.0, 1.0)

def apply_log_to_rec709_cst(bgr_float: np.ndarray, black_floor: float = 0.11, white_ceil: float = 0.95) -> np.ndarray:
    img = np.clip((bgr_float - black_floor) / max(0.1, white_ceil - black_floor), 0.0, 1.0)
    
    # Sigmoidal S-curve mapping for Log-to-Rec.709 expansion
    p = 0.40
    c = 1.35
    below = p * (np.maximum(0.0, img / p) ** c)
    above = 1.0 - (1.0 - p) * (np.maximum(0.0, (1.0 - img) / (1.0 - p)) ** c)
    img = np.where(img < p, below, above)
    
    # Pure float32 HSV saturation expansion to Rec.709
    hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.65, 0.0, 1.0)
    rec709_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return np.clip(rec709_bgr, 0.0, 1.0)

def apply_input_camera_profile_with_diagnostics(
    bgr_float: np.ndarray,
    profile: str = "rec709",
    black_floor: float = 0.11,
    white_ceil: float = 0.95,
    probed_info: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, NormalizationDiagnostics]:
    p = profile.lower().strip()
    if p == "auto_ask":
        raise ValueError("auto_ask is a pending decision state and cannot be applied as a camera profile transform.")

    src_range = str(probed_info.get("color_range", "tv")) if probed_info else "tv"
    src_transfer = str(probed_info.get("color_transfer", "unknown")) if probed_info else "unknown"

    # Pre-transform source statistics
    src_rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
    src_lum = 0.2126 * src_rgb[..., 0] + 0.7152 * src_rgb[..., 1] + 0.0722 * src_rgb[..., 2]
    src_p5 = float(np.percentile(src_lum, 5) * 255.0)
    src_p25 = float(np.percentile(src_lum, 25) * 255.0)
    src_p50 = float(np.percentile(src_lum, 50) * 255.0)
    src_p75 = float(np.percentile(src_lum, 75) * 255.0)
    src_p95 = float(np.percentile(src_lum, 95) * 255.0)
    src_iqr = src_p75 - src_p25

    expected_black = 0.0
    neg_pct = 0.0
    over_one_pct = 0.0

    if p in ["rec709", "bt709", "srgb", "display"]:
        expected_black = 0.0
        resolved_p = "rec709"
        norm_bgr = np.clip(bgr_float, 0.0, 1.0)
        display_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)
        pos_collapsed_pct = 0.0
        nonpos_pct = float(np.mean(src_lum <= 0.001) * 100.0)

    elif "dlog_m" in p or "dlog-m" in p or "dji_dlog_m" in p:
        expected_black = 0.10
        resolved_p = "dji_dlog_m_rec709"
        rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
        linear_rgb = dji_dlog_m_to_linear(rgb)
        lin_lum = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
        nonpos_pct = float(np.mean(lin_lum <= 0.001) * 100.0)
        pos_mask = lin_lum > 0.015

        compressed, neg_pct, _ = compress_out_of_gamut_rgb(linear_rgb)
        over_one_pct = float(np.mean(np.max(compressed, axis=-1) > 1.0) * 100.0)
        display_rgb = scene_linear_to_rec709_display(compressed)
        
        # Saturation normalization (1.30x)
        hsv = cv2.cvtColor(np.clip(display_rgb, 0.0, 1.0).astype(np.float32), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.30, 0.0, 1.0)
        norm_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        display_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)

        disp_lum = 0.2126 * display_rgb[..., 0] + 0.7152 * display_rgb[..., 1] + 0.0722 * display_rgb[..., 2]
        if np.any(pos_mask):
            pos_collapsed_pct = float(np.mean((disp_lum < (2.0 / 255.0)) & pos_mask) * 100.0)
        else:
            pos_collapsed_pct = 0.0

    elif "dlog" in p or "d_log" in p or "d-log" in p:
        expected_black = 0.0929
        resolved_p = "dji_dlog_dgamut"
        rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
        linear_rgb = dji_dlog_to_linear(rgb)
        lin_lum = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
        nonpos_pct = float(np.mean(lin_lum <= 0.001) * 100.0)
        pos_mask = lin_lum > 0.015

        h, w, c_dim = linear_rgb.shape
        reshaped = linear_rgb.reshape(-1, 3)
        converted = np.dot(reshaped, MAT_DGAMUT_TO_BT709.T).reshape(h, w, c_dim)
        compressed, neg_pct, _ = compress_out_of_gamut_rgb(converted)
        over_one_pct = float(np.mean(np.max(compressed, axis=-1) > 1.0) * 100.0)
        display_rgb = scene_linear_to_rec709_display(compressed)
        norm_bgr = cv2.cvtColor(display_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)

        disp_lum = 0.2126 * display_rgb[..., 0] + 0.7152 * display_rgb[..., 1] + 0.0722 * display_rgb[..., 2]
        if np.any(pos_mask):
            pos_collapsed_pct = float(np.mean((disp_lum < (2.0 / 255.0)) & pos_mask) * 100.0)
        else:
            pos_collapsed_pct = 0.0

    elif "slog3" in p or "s_log3" in p:
        expected_black = 0.09286
        resolved_p = "sony_slog3_sgamut3cine"
        rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
        linear_rgb = sony_slog3_to_linear(rgb)
        lin_lum = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
        nonpos_pct = float(np.mean(lin_lum <= 0.001) * 100.0)
        pos_mask = lin_lum > 0.015

        h, w, c_dim = linear_rgb.shape
        reshaped = linear_rgb.reshape(-1, 3)
        converted = np.dot(reshaped, MAT_SGAMUT3CINE_TO_BT709.T).reshape(h, w, c_dim)
        compressed, neg_pct, _ = compress_out_of_gamut_rgb(converted)
        over_one_pct = float(np.mean(np.max(compressed, axis=-1) > 1.0) * 100.0)
        display_rgb = scene_linear_to_rec709_display(compressed)
        norm_bgr = cv2.cvtColor(display_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)

        disp_lum = 0.2126 * display_rgb[..., 0] + 0.7152 * display_rgb[..., 1] + 0.0722 * display_rgb[..., 2]
        if np.any(pos_mask):
            pos_collapsed_pct = float(np.mean((disp_lum < (2.0 / 255.0)) & pos_mask) * 100.0)
        else:
            pos_collapsed_pct = 0.0

    elif "apple_log" in p or "apple" in p:
        expected_black = 0.15048
        resolved_p = "apple_log_rec2020"
        rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
        linear_rgb = apple_log_to_linear(rgb)
        lin_lum = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
        nonpos_pct = float(np.mean(lin_lum <= 0.001) * 100.0)
        pos_mask = lin_lum > 0.015

        h, w, c_dim = linear_rgb.shape
        reshaped = linear_rgb.reshape(-1, 3)
        converted = np.dot(reshaped, MAT_BT2020_TO_BT709.T).reshape(h, w, c_dim)
        compressed, neg_pct, _ = compress_out_of_gamut_rgb(converted)
        over_one_pct = float(np.mean(np.max(compressed, axis=-1) > 1.0) * 100.0)
        display_rgb = scene_linear_to_rec709_display(compressed)
        norm_bgr = cv2.cvtColor(display_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)

        disp_lum = 0.2126 * display_rgb[..., 0] + 0.7152 * display_rgb[..., 1] + 0.0722 * display_rgb[..., 2]
        if np.any(pos_mask):
            pos_collapsed_pct = float(np.mean((disp_lum < (2.0 / 255.0)) & pos_mask) * 100.0)
        else:
            pos_collapsed_pct = 0.0

    elif p in ["generic_log_experimental", "generic_log", "generic log", "flat", "log"]:
        expected_black = black_floor
        resolved_p = "generic_log_experimental"
        norm_bgr = apply_log_to_rec709_cst(bgr_float, black_floor=black_floor, white_ceil=white_ceil)
        display_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)
        pos_collapsed_pct = 0.0
        nonpos_pct = float(np.mean(src_lum <= black_floor) * 100.0)

    else:
        raise ValueError(f"Unknown camera input profile '{profile}'. Supported profiles: rec709, sony_slog3_sgamut3cine, apple_log_rec2020, dji_dlog_dgamut, dji_dlog_m_rec709, generic_log_experimental.")

    # Post-normalization statistics
    post_lum = 0.2126 * display_rgb[..., 0] + 0.7152 * display_rgb[..., 1] + 0.0722 * display_rgb[..., 2]
    norm_p5 = float(np.percentile(post_lum, 5) * 255.0)
    norm_p25 = float(np.percentile(post_lum, 25) * 255.0)
    norm_p50 = float(np.percentile(post_lum, 50) * 255.0)
    norm_p75 = float(np.percentile(post_lum, 75) * 255.0)
    norm_p95 = float(np.percentile(post_lum, 95) * 255.0)
    norm_iqr = norm_p75 - norm_p25

    post_black_pct = float(np.mean(post_lum < (2.0 / 255.0)) * 100.0)
    post_highlight_pct = float(np.mean(post_lum > (253.0 / 255.0)) * 100.0)
    finite_passed = bool(np.all(np.isfinite(norm_bgr)))

    diagnostics = NormalizationDiagnostics(
        requested_profile=profile,
        resolved_profile=resolved_p,
        source_color_range=src_range,
        source_transfer_metadata=src_transfer,
        expected_log_black_code=round(expected_black, 4),
        decoded_nonpositive_luminance_pct=round(nonpos_pct, 2),
        positive_luminance_collapsed_to_black_pct=round(pos_collapsed_pct, 2),
        negative_rgb_excursion_pct=round(neg_pct, 2),
        over_one_rgb_excursion_pct=round(over_one_pct, 2),
        post_black_occupancy_pct=round(post_black_pct, 2),
        post_highlight_occupancy_pct=round(post_highlight_pct, 2),
        source_p5=round(src_p5, 1),
        source_p25=round(src_p25, 1),
        source_p50=round(src_p50, 1),
        source_p75=round(src_p75, 1),
        source_p95=round(src_p95, 1),
        normalized_p5=round(norm_p5, 1),
        normalized_p25=round(norm_p25, 1),
        normalized_p50=round(norm_p50, 1),
        normalized_p75=round(norm_p75, 1),
        normalized_p95=round(norm_p95, 1),
        source_iqr=round(src_iqr, 1),
        normalized_iqr=round(norm_iqr, 1),
        finite_values_passed=finite_passed
    )
    return np.clip(norm_bgr, 0.0, 1.0), diagnostics

def apply_input_camera_profile(
    bgr_float: np.ndarray,
    profile: str = "rec709",
    black_floor: float = 0.11,
    white_ceil: float = 0.95
) -> np.ndarray:
    norm_bgr, _ = apply_input_camera_profile_with_diagnostics(
        bgr_float=bgr_float,
        profile=profile,
        black_floor=black_floor,
        white_ceil=white_ceil
    )
    return norm_bgr

def calculate_deterministic_match_params(
    reference: ShotMetrics,
    target: ShotMetrics,
    strength: float = 1.0
) -> ColorGradeParams:
    ref_l_mean, ref_a_mean, ref_b_mean = reference.avg_lab_mean
    ref_l_std, ref_a_std, ref_b_std = reference.avg_lab_std
    
    tgt_l_mean, tgt_a_mean, tgt_b_mean = target.avg_lab_mean
    tgt_l_std, tgt_a_std, tgt_b_std = target.avg_lab_std
    
    if ref_l_std < 1.0 or tgt_l_std < 1.0:
        l_gain = 1.0
    else:
        l_gain = float(np.clip(ref_l_std / tgt_l_std, 0.7, 1.5))
        
    if ref_a_std < 1.0 or tgt_a_std < 1.0:
        a_gain = 1.0
    else:
        a_gain = float(np.clip(ref_a_std / tgt_a_std, 0.6, 1.6))
        
    if ref_b_std < 1.0 or tgt_b_std < 1.0:
        b_gain = 1.0
    else:
        b_gain = float(np.clip(ref_b_std / tgt_b_std, 0.6, 1.6))
        
    raw_l_offset = float(ref_l_mean - l_gain * tgt_l_mean)
    a_offset = float(ref_a_mean - a_gain * tgt_a_mean)
    b_offset = float(ref_b_mean - b_gain * tgt_b_mean)

    # Composition-Aware Luminance Protection:
    # Check if endpoints (p5 and p95) are already aligned while mean differs due to framing/area proportions
    ref_p5 = reference.p5_luminance / 2.55 if reference.p5_luminance > 0 else (reference.sampled_frames[0].p5_luminance / 2.55 if reference.sampled_frames else 0.0)
    ref_p95 = reference.p95_luminance / 2.55 if reference.p95_luminance > 0 else (reference.sampled_frames[0].p95_luminance / 2.55 if reference.sampled_frames else 100.0)
    tgt_p5 = target.p5_luminance / 2.55 if target.p5_luminance > 0 else (target.sampled_frames[0].p5_luminance / 2.55 if target.sampled_frames else 0.0)
    tgt_p95 = target.p95_luminance / 2.55 if target.p95_luminance > 0 else (target.sampled_frames[0].p95_luminance / 2.55 if target.sampled_frames else 100.0)

    p5_delta = ref_p5 - tgt_p5
    p95_delta = ref_p95 - tgt_p95
    endpoint_delta = 0.5 * (p5_delta + p95_delta)

    comp_disparity = abs(raw_l_offset - endpoint_delta)
    if comp_disparity > 10.0 and abs(p5_delta) < 15.0 and abs(p95_delta) < 15.0:
        # Tonal extremes are already lit similarly; area proportions dominate the mean.
        # Anchor predominantly on endpoint delta to avoid false composition exposure lift
        dampen = max(0.08, 1.0 - (comp_disparity - 10.0) / 15.0)
        l_offset = endpoint_delta + (raw_l_offset - endpoint_delta) * dampen
    else:
        l_offset = raw_l_offset

    # Headroom-aware clipping guard: do not push highlights into blown-out clipping
    max_safe_lift = max(0.0, 94.0 - tgt_p95)
    max_safe_drop = max(0.0, tgt_p5 - 2.0)
    l_offset = float(np.clip(l_offset, -max_safe_drop, max_safe_lift))
    
    l_gain = 1.0 + (l_gain - 1.0) * strength
    a_gain = 1.0 + (a_gain - 1.0) * strength
    b_gain = 1.0 + (b_gain - 1.0) * strength
    l_offset = float(np.clip(l_offset * strength, -80.0, 80.0))
    a_offset = float(np.clip(a_offset * strength, -80.0, 80.0))
    b_offset = float(np.clip(b_offset * strength, -80.0, 80.0))
    
    return ColorGradeParams(
        exposure_ev=0.0,
        contrast=1.0,
        pivot=0.45,
        saturation=1.0,
        temperature=0.0,
        tint=0.0,
        lab_l_gain=round(l_gain, 3),
        lab_l_offset=round(l_offset, 3),
        lab_a_gain=round(a_gain, 3),
        lab_a_offset=round(a_offset, 3),
        lab_b_gain=round(b_gain, 3),
        lab_b_offset=round(b_offset, 3)
    )

def apply_color_grade_to_frame(
    bgr_frame: np.ndarray,
    plan_or_params: Union[GradePlan, ColorGradeParams],
    is_log: bool = False
) -> np.ndarray:
    # Pure 32-bit Floating-point representation in [0.0, 1.0]
    if bgr_frame.dtype == np.uint8:
        img = bgr_frame.astype(np.float32) / 255.0
    else:
        img = bgr_frame.astype(np.float32).copy()
        if np.max(img) > 1.0:
            img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    
    # Extract staged parameters
    if isinstance(plan_or_params, GradePlan):
        plan = plan_or_params
        do_log = plan.input_transform.is_log or is_log
        log_floor = plan.input_transform.black_floor
        log_ceil = plan.input_transform.white_ceil
        
        exposure_ev = plan.technical_balance.exposure_ev + plan.scene_trim.trim_exposure_ev
        temp = plan.technical_balance.temperature
        tint = plan.technical_balance.tint
        
        l_gain = plan.scene_match.lab_l_gain
        l_offset = plan.scene_match.lab_l_offset
        a_gain = plan.scene_match.lab_a_gain
        a_offset = plan.scene_match.lab_a_offset
        b_gain = plan.scene_match.lab_b_gain
        b_offset = plan.scene_match.lab_b_offset
        
        contrast = plan.creative_look.contrast * plan.scene_trim.trim_contrast
        pivot = plan.creative_look.pivot
        saturation = plan.creative_look.saturation * plan.scene_trim.trim_saturation
        shadow_sat_trim = getattr(plan.scene_trim, "trim_shadow_sat", 1.0)
        shadow_bias = plan.creative_look.shadow_rgb_offset
        highlight_bias = plan.creative_look.highlight_rgb_offset
        black_toe_lift = plan.creative_look.black_toe_lift + plan.scene_trim.trim_shadow_lift
        
        shoulder_thresh = plan.output_transform.highlight_shoulder_threshold
        compression_factor = plan.output_transform.highlight_compression_factor
    else:
        params = plan_or_params
        do_log = is_log
        log_floor, log_ceil = 0.11, 0.95
        
        exposure_ev = params.exposure_ev
        temp = params.temperature
        tint = params.tint
        
        l_gain = params.lab_l_gain
        l_offset = params.lab_l_offset
        a_gain = params.lab_a_gain
        a_offset = params.lab_a_offset
        b_gain = params.lab_b_gain
        b_offset = params.lab_b_offset
        
        contrast = params.contrast
        pivot = params.pivot
        saturation = params.saturation
        shadow_bias = params.shadow_rgb_offset
        highlight_bias = params.highlight_rgb_offset
        black_toe_lift = 0.0
        shadow_sat_trim = 1.0
        
        shoulder_thresh = 0.85
        compression_factor = 2.0

    # STAGE 1: Input Transform / Camera Profile Normalization
    profile = "rec709"
    if isinstance(plan_or_params, GradePlan):
        profile = plan_or_params.input_transform.profile
        if profile == "auto_ask":
            raise ValueError("auto_ask is a pending decision state and cannot be applied in apply_color_grade_to_frame. A concrete profile must be resolved before grading.")
        if profile == "rec709" and (plan_or_params.input_transform.is_log or is_log):
            profile = "generic_log_experimental"
    elif is_log:
        profile = "generic_log_experimental"
        
    if profile != "rec709":
        img = apply_input_camera_profile(img, profile=profile, black_floor=log_floor, white_ceil=log_ceil)
        
    # STAGE 2: Per-Shot Technical Balance (Primary Exposure & White Balance)
    if abs(exposure_ev) > 0.001:
        img = img * float(2.0 ** exposure_ev)
        
    if abs(temp) > 0.1 or abs(tint) > 0.1:
        temp_f = temp / 150.0
        tint_f = tint / 150.0
        img[:, :, 2] = img[:, :, 2] * (1.0 + temp_f) # Red
        img[:, :, 0] = img[:, :, 0] * (1.0 - temp_f) # Blue
        img[:, :, 1] = img[:, :, 1] * (1.0 - tint_f) # Green
        
    img = np.clip(img, 0.0, 1.0)
    
    # STAGE 3: Same-Scene Shot Match (Pure Float32 CIELAB Perceptual Alignment)
    if (abs(l_gain - 1.0) > 0.001 or abs(l_offset) > 0.001 or
        abs(a_gain - 1.0) > 0.001 or abs(a_offset) > 0.001 or
        abs(b_gain - 1.0) > 0.001 or abs(b_offset) > 0.001):
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # Float32 LAB
        
        l = lab[:, :, 0] # L in [0, 100]
        a = lab[:, :, 1] # a in [-127, 127]
        b = lab[:, :, 2] # b in [-127, 127]
        
        lab[:, :, 0] = np.clip(l_gain * l + l_offset, 0.0, 100.0)
        lab[:, :, 1] = np.clip(a_gain * a + a_offset, -127.0, 127.0)
        lab[:, :, 2] = np.clip(b_gain * b + b_offset, -127.0, 127.0)
        
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # STAGE 4: Shared Creative Look (Filmic Contrast, Highlight/Shadow Tints, Saturation)
    # A. Filmic S-Curve Contrast
    if abs(contrast - 1.0) > 0.005:
        p = float(np.clip(pivot, 0.05, 0.95))
        c = float(contrast)
        x_norm = np.clip(img, 0.0, 1.0)
        below = p * (np.maximum(0.0, x_norm / p) ** c)
        above = 1.0 - (1.0 - p) * (np.maximum(0.0, (1.0 - x_norm) / (1.0 - p)) ** c)
        img = np.where(x_norm < p, below, above)

    # B. Luminance-Weighted Highlight and Shadow Biases (Split Toning in Float Space)
    lum = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    w_highlight = np.clip((lum - 0.55) / 0.40, 0.0, 1.0) ** 1.5
    w_shadow = np.clip((0.45 - lum) / 0.40, 0.0, 1.0) ** 1.5
    
    if highlight_bias and any(abs(x) > 0.001 for x in highlight_bias):
        img[:, :, 0] += highlight_bias[0] * w_highlight
        img[:, :, 1] += highlight_bias[1] * w_highlight
        img[:, :, 2] += highlight_bias[2] * w_highlight
        
    if shadow_bias and any(abs(x) > 0.001 for x in shadow_bias):
        img[:, :, 0] += shadow_bias[0] * w_shadow
        img[:, :, 1] += shadow_bias[1] * w_shadow
        img[:, :, 2] += shadow_bias[2] * w_shadow
        
    # C. Black-Mist-Inspired Tonal Response (Shadow Toe Lift)
    if abs(black_toe_lift) > 0.01:
        toe_f = black_toe_lift / 255.0
        toe_weight = np.clip((0.40 - lum) / 0.40, 0.0, 1.0) ** 2.0
        img += toe_f * np.expand_dims(toe_weight, axis=-1)

    img = np.clip(img, 0.0, 1.0)

    # D. Saturation in Pure Float32 HSV space with Luminance-Zone Awareness & Clipping Guard
    if abs(saturation - 1.0) > 0.005 or abs(shadow_sat_trim - 1.0) > 0.005:
        hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        
        # Pixel luminance for zone-specific saturation control
        pix_lum = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
        
        # Deep shadow saturation dampening (pix_lum < 0.25): prevent noise blowout and oversaturated darks
        if saturation > 1.0:
            shadow_dampen = np.clip(pix_lum / 0.25, 0.65, 1.0)
            eff_sat = 1.0 + (saturation - 1.0) * shadow_dampen
        else:
            eff_sat = saturation
            
        # Selective shadow-region saturation damping
        if abs(shadow_sat_trim - 1.0) > 0.005:
            shadow_zone_weight = np.clip((0.30 - pix_lum) / 0.30, 0.0, 1.0)
            eff_sat = eff_sat * (1.0 - shadow_zone_weight * (1.0 - shadow_sat_trim))

        sat_channel = sat_channel * eff_sat
        
        # RGB channel clipping guard: dampen saturation boost as any channel approaches clipping (>0.94)
        max_ch = np.maximum(np.maximum(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
        clip_guard = np.where(max_ch > 0.94, np.clip((1.0 - max_ch) / 0.06, 0.4, 1.0), 1.0)
        sat_channel = np.clip(sat_channel * clip_guard, 0.0, 1.0)
        
        hsv[:, :, 1] = sat_channel
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # STAGE 6: Output Transform (Highlight Shoulder Compression & Clipping Guard)
    img = np.where(
        img > shoulder_thresh,
        shoulder_thresh + (img - shoulder_thresh) / (1.0 + (img - shoulder_thresh) * compression_factor),
        img
    )
    img = np.clip(img, 0.0, 1.0)
    
    if bgr_frame.dtype == np.uint8:
        return (img * 255.0).astype(np.uint8)
    return img

def create_creative_probe_set() -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    """Generates standardized synthetic probes to isolate shared creative look invariants."""
    ramp_vals = [25, 64, 128, 191, 230]
    ramp = [np.full((10, 10, 3), v, dtype=np.uint8) for v in ramp_vals]
    color_patches = {
        "warm": np.full((10, 10, 3), [70, 120, 180], dtype=np.uint8),  # BGR
        "cool": np.full((10, 10, 3), [180, 120, 70], dtype=np.uint8),
        "red": np.full((10, 10, 3), [30, 30, 200], dtype=np.uint8),
        "green": np.full((10, 10, 3), [30, 200, 30], dtype=np.uint8),
        "blue": np.full((10, 10, 3), [200, 30, 30], dtype=np.uint8),
        "shadow_neutral": np.full((10, 10, 3), 35, dtype=np.uint8),
        "highlight_neutral": np.full((10, 10, 3), 215, dtype=np.uint8)
    }
    return ramp, color_patches

def evaluate_transform_look_continuity(
    ref_plan: Optional[GradePlan],
    cand_plan: Optional[GradePlan],
    candidate_metrics: Optional[ShotMetrics] = None
) -> LookContinuityScore:
    """Evaluates cross-scene look continuity by testing creative-only plans on standardized probes.
    Excludes input transforms, per-shot technical balance, scene matching, and scene trims.
    """
    if cand_plan is None:
        return LookContinuityScore(
            overall_score=0.0,
            contrast_slope_adherence=0.0,
            highlight_split_adherence=0.0,
            shadow_split_adherence=0.0,
            saturation_scaling_adherence=0.0,
            probe_tonal_continuity=0.0,
            probe_chromatic_harmony=0.0,
            passed=False,
            diagnosis="Ungraded baseline: creative look transform not yet applied."
        )

    if ref_plan is None:
        return LookContinuityScore(
            overall_score=100.0,
            contrast_slope_adherence=100.0,
            highlight_split_adherence=100.0,
            shadow_split_adherence=100.0,
            saturation_scaling_adherence=100.0,
            probe_tonal_continuity=100.0,
            probe_chromatic_harmony=100.0,
            passed=True,
            diagnosis="Master reference look baseline established."
        )

    # Isolate creative look ONLY (excluding input transform, technical balance, scene match, scene trim)
    clean_ref = GradePlan(shot_id="clean_ref", creative_look=ref_plan.creative_look)
    clean_cand = GradePlan(shot_id="clean_cand", creative_look=cand_plan.creative_look)
    
    ramp, patches = create_creative_probe_set()
    diagnosis = []
    
    # 1. Grayscale ramp midtone contrast curve slope (isolated from per-shot exposure/trims)
    ref_ramp_out = [apply_color_grade_to_frame(f, clean_ref) for f in ramp]
    cand_ramp_out = [apply_color_grade_to_frame(f, clean_cand) for f in ramp]
    
    ref_ramp_l = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0])) for f in ref_ramp_out]
    cand_ramp_l = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0])) for f in cand_ramp_out]
    
    # Midtone slope across p75-p25: (191 - 64)
    ref_slope = (ref_ramp_l[3] - ref_ramp_l[1]) / 127.0
    cand_slope = (cand_ramp_l[3] - cand_ramp_l[1]) / 127.0
    slope_diff = abs(ref_slope - cand_slope)
    contrast_score = float(100.0 * np.exp(-slope_diff / 0.15))
    if contrast_score < 70.0:
        diagnosis.append(f"Creative contrast slope divergence: ref={ref_slope:.2f}, cand={cand_slope:.2f}")
        
    # 2. Saturation scaling on chromatic color patches (warm, cool, red, green, blue)
    ref_patch_out = {k: apply_color_grade_to_frame(v, clean_ref) for k, v in patches.items()}
    cand_patch_out = {k: apply_color_grade_to_frame(v, clean_cand) for k, v in patches.items()}
    
    chroma_diffs = []
    for k in ["warm", "cool", "red", "green", "blue"]:
        ref_lab = cv2.cvtColor(ref_patch_out[k], cv2.COLOR_BGR2LAB).astype(np.float32)
        cand_lab = cv2.cvtColor(cand_patch_out[k], cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_c = np.mean(np.sqrt((ref_lab[:, :, 1] - 128.0)**2 + (ref_lab[:, :, 2] - 128.0)**2))
        cand_c = np.mean(np.sqrt((cand_lab[:, :, 1] - 128.0)**2 + (cand_lab[:, :, 2] - 128.0)**2))
        chroma_diffs.append(abs(ref_c - cand_c))
        
    mean_chroma_diff = float(np.mean(chroma_diffs))
    sat_score = float(100.0 * np.exp(-mean_chroma_diff / 8.0))
    if sat_score < 70.0:
        diagnosis.append(f"Creative saturation scaling mismatch: delta={mean_chroma_diff:.1f}")
        
    # 3. Highlight and Shadow split-toning adherence
    ref_hl_lab = cv2.cvtColor(ref_patch_out["highlight_neutral"], cv2.COLOR_BGR2LAB).astype(np.float32)
    cand_hl_lab = cv2.cvtColor(cand_patch_out["highlight_neutral"], cv2.COLOR_BGR2LAB).astype(np.float32)
    hl_delta = float(np.sqrt(np.mean((ref_hl_lab[:, :, 1:] - cand_hl_lab[:, :, 1:])**2)))
    hl_score = float(100.0 * np.exp(-hl_delta / 6.0))
    if hl_score < 70.0:
        diagnosis.append(f"Highlight split-tone divergence: delta={hl_delta:.1f}")
        
    ref_sh_lab = cv2.cvtColor(ref_patch_out["shadow_neutral"], cv2.COLOR_BGR2LAB).astype(np.float32)
    cand_sh_lab = cv2.cvtColor(cand_patch_out["shadow_neutral"], cv2.COLOR_BGR2LAB).astype(np.float32)
    sh_delta = float(np.sqrt(np.mean((ref_sh_lab[:, :, 1:] - cand_sh_lab[:, :, 1:])**2)))
    sh_score = float(100.0 * np.exp(-sh_delta / 6.0))
    if sh_score < 70.0:
        diagnosis.append(f"Shadow split-tone divergence: delta={sh_delta:.1f}")
        
    # Tonal continuity combines contrast curve adherence and highlight split adherence
    tonal_continuity = 0.60 * contrast_score + 0.40 * hl_score
    # Chromatic harmony combines saturation scaling and shadow split adherence
    chromatic_harmony = 0.60 * sat_score + 0.40 * sh_score
    
    overall = float(np.clip(0.50 * tonal_continuity + 0.50 * chromatic_harmony, 0.0, 100.0))
    passed = (overall >= 75.0)
    
    diag_str = "; ".join(diagnosis) if diagnosis else "Creative look invariants harmonized across probes."

    return LookContinuityScore(
        overall_score=round(overall, 1),
        contrast_slope_adherence=round(contrast_score, 1),
        highlight_split_adherence=round(hl_score, 1),
        shadow_split_adherence=round(sh_score, 1),
        saturation_scaling_adherence=round(sat_score, 1),
        probe_tonal_continuity=round(tonal_continuity, 1),
        probe_chromatic_harmony=round(chromatic_harmony, 1),
        passed=passed,
        diagnosis=diag_str
    )

def evaluate_scene_health(
    source_metrics: ShotMetrics,
    graded_metrics: ShotMetrics,
    graded_frames: Optional[List[np.ndarray]] = None,
    scene_intent: Optional[SceneIntent] = None
) -> SceneHealthScore:
    """Evaluates objective image health and scene-intent compliance for a graded shot.
    Enforces hard gates on excessive clipping, shadow oversaturation, and midtone crush.
    """
    hard_gate_failures = []
    
    # 1. Source-Relative Exposure Change & Appropriateness
    src_lum = max(1.0, source_metrics.avg_luminance)
    grd_lum = max(1.0, graded_metrics.avg_luminance)
    exposure_ev_change = float(np.log2(grd_lum / src_lum))
    
    if scene_intent:
        min_ev, max_ev = scene_intent.source_relative_exposure_bounds
    else:
        min_ev, max_ev = -1.2, 1.2
        
    if min_ev <= exposure_ev_change <= max_ev:
        exp_score = 100.0
    elif exposure_ev_change < min_ev:
        exp_score = max(0.0, 100.0 - (min_ev - exposure_ev_change) * 35.0)
    else:
        exp_score = max(0.0, 100.0 - (exposure_ev_change - max_ev) * 35.0)
        
    # 2. Midtone Readability
    src_p50 = source_metrics.p50_luminance if source_metrics.p50_luminance > 0 else source_metrics.avg_luminance
    grd_p50 = graded_metrics.p50_luminance if graded_metrics.p50_luminance > 0 else graded_metrics.avg_luminance
    grd_iqr = graded_metrics.p75_luminance - graded_metrics.p25_luminance
    src_iqr = source_metrics.p75_luminance - source_metrics.p25_luminance
    
    lighting = scene_intent.lighting_class if scene_intent else "daylight"
    exp_intent = scene_intent.exposure_class if scene_intent else "balanced"
    is_silhouette = (lighting == "intentional_silhouette" or exp_intent == "intentional_silhouette")
    
    if is_silhouette:
        mid_score = 100.0
    elif lighting in ["low_key_night", "practical_night"] or exp_intent in ["low_key", "low_key_underexposed"]:
        if src_p50 >= 20.0 and grd_p50 < 15.0:
            mid_score = max(20.0, 100.0 - (15.0 - grd_p50) * 8.0)
        else:
            mid_score = 100.0
    else:
        if src_p50 >= 30.0 and grd_p50 < 18.0:
            mid_score = max(20.0, 100.0 - (18.0 - grd_p50) * 6.0)
        else:
            mid_score = 100.0
            
    if not is_silhouette and grd_iqr < 18.0 and src_iqr > 35.0:
        mid_score = min(mid_score, max(20.0, 100.0 - (18.0 - grd_iqr) * 4.0))
        
    # Hard Gate 3: Midtone crush (source had midtones > 25, graded fell below 10)
    if not is_silhouette and src_p50 > 25.0 and grd_p50 < 10.0:
        hard_gate_failures.append(f"Midtone crush: midtones collapsed below readable floor (source p50={src_p50:.1f}, graded p50={grd_p50:.1f})")

    # 3. Clipping Health (Advisory - clipping gates removed per user request)
    sh_clip = graded_metrics.avg_shadow_clip_pct if graded_metrics.avg_shadow_clip_pct > 0 else 0.0
    hl_clip = graded_metrics.avg_highlight_clip_pct if graded_metrics.avg_highlight_clip_pct > 0 else 0.0
    clipping_health_score = 100.0
        
    # 4. Shadow and Midtone Saturation Health (Normalized HSV [0.0, 1.0] Space)
    mean_shadow_sat = 0.0
    mean_mid_sat = 0.0
    mean_hl_sat = 0.0
    if graded_frames and len(graded_frames) > 0:
        shadow_sats = []
        mid_sats = []
        hl_sats = []
        for f in graded_frames:
            f_bgr = f if f.dtype == np.uint8 else (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2HSV)
            s_chan = hsv[:, :, 1].astype(np.float32) / 255.0
            sh_mask = gray < 45
            mid_mask = (gray >= 45) & (gray <= 200)
            hl_mask = gray > 200
            if np.any(sh_mask):
                shadow_sats.append(float(np.mean(s_chan[sh_mask])))
            if np.any(mid_mask):
                mid_sats.append(float(np.mean(s_chan[mid_mask])))
            if np.any(hl_mask):
                hl_sats.append(float(np.mean(s_chan[hl_mask])))
        if shadow_sats:
            mean_shadow_sat = float(np.mean(shadow_sats))
        if mid_sats:
            mean_mid_sat = float(np.mean(mid_sats))
        if hl_sats:
            mean_hl_sat = float(np.mean(hl_sats))
    else:
        mean_shadow_sat = float(np.clip(graded_metrics.avg_chroma * 0.015, 0.0, 1.0))
        mean_mid_sat = float(np.clip(graded_metrics.avg_chroma * 0.020, 0.0, 1.0))
        mean_hl_sat = float(np.clip(graded_metrics.avg_chroma * 0.010, 0.0, 1.0))

    # Read ceilings from SceneIntent (or defaults in normalized [0, 1] HSV space)
    raw_sh_ceil = getattr(scene_intent, "shadow_output_chroma_ceiling", None)
    if raw_sh_ceil is None:
        raw_sh_ceil = getattr(scene_intent, "shadow_saturation_ceiling", 0.35)
        if raw_sh_ceil > 1.0:
            raw_sh_ceil = raw_sh_ceil / 255.0
    raw_mid_ceil = getattr(scene_intent, "midtone_output_chroma_ceiling", 0.70)
    if raw_mid_ceil > 1.0:
        raw_mid_ceil = raw_mid_ceil / 255.0

    # Deterministic hard policy bounds: model cannot widen health thresholds beyond safe limits
    is_dark_scene = (grd_p50 < 25.0 or (scene_intent and scene_intent.lighting_class in ["low_key_night", "night", "practical_night"]))
    max_safe_sh_ceiling = 0.40 if is_dark_scene else 0.55
    effective_sh_ceil = min(max_safe_sh_ceiling, float(raw_sh_ceil))
    effective_mid_ceil = min(0.75, float(raw_mid_ceil))

    if mean_shadow_sat > effective_sh_ceil:
        sat_pen = min(80.0, (mean_shadow_sat - effective_sh_ceil) / max(0.1, 1.0 - effective_sh_ceil) * 120.0)
        shadow_sat_score = max(0.0, 100.0 - sat_pen)
    else:
        shadow_sat_score = 100.0

    # Hard Gate 2: Shadow oversaturation in dark scenes
    if grd_p50 < 25.0 and (mean_shadow_sat > 0.40 or (graded_metrics.avg_chroma > 25.0 and mean_shadow_sat > 0.35)):
        hard_gate_failures.append(f"Severe shadow oversaturation in dark scene (median lum={grd_p50:.1f}, shadow sat={mean_shadow_sat:.2f} > {effective_sh_ceil:.2f})")

    # Midtone chroma health
    if mean_mid_sat > effective_mid_ceil:
        mid_pen = min(50.0, (mean_mid_sat - effective_mid_ceil) / max(0.1, 1.0 - effective_mid_ceil) * 100.0)
        midtone_chroma_score = max(30.0, 100.0 - mid_pen)
        if mean_mid_sat > 0.85:
            hard_gate_failures.append(f"Excessive midtone saturation (midtone sat={mean_mid_sat:.2f} > 0.85)")
    else:
        midtone_chroma_score = 100.0

    # 5. Highlight Chroma Health
    if mean_hl_sat > 0.45:
        hl_pen = min(60.0, (mean_hl_sat - 0.45) * 100.0)
        hl_chroma_score = max(20.0, 100.0 - hl_pen)
    else:
        hl_chroma_score = 100.0
        
    hard_gates_passed = (len(hard_gate_failures) == 0)
    raw_overall = (
        0.20 * exp_score +
        0.20 * mid_score +
        0.20 * clipping_health_score +
        0.20 * shadow_sat_score +
        0.10 * midtone_chroma_score +
        0.10 * hl_chroma_score
    )
    if not hard_gates_passed:
        overall_health = float(min(45.0, raw_overall))
        passed = False
    else:
        overall_health = float(raw_overall)
        passed = (overall_health >= 70.0)
        
    diag_parts = []
    if hard_gate_failures:
        diag_parts.extend(hard_gate_failures)
    elif not passed:
        if exp_score < 70.0:
            diag_parts.append(f"Exposure change ({exposure_ev_change:+.2f} EV) exceeds scene intent bounds")
        if mid_score < 70.0:
            diag_parts.append(f"Midtone readability compromised (p50: {grd_p50:.1f})")
        if clipping_health_score < 70.0:
            diag_parts.append(f"Shadow/highlight clipping elevated")
        if shadow_sat_score < 70.0:
            diag_parts.append(f"Deep shadow saturation ({mean_shadow_sat:.2f}) exceeds scene ceiling ({effective_sh_ceil:.2f})")
        if midtone_chroma_score < 70.0:
            diag_parts.append(f"Midtone saturation ({mean_mid_sat:.2f}) exceeds ceiling ({effective_mid_ceil:.2f})")
    else:
        diag_parts.append("Scene image health verified within target parameters.")

    return SceneHealthScore(
        overall_score=round(overall_health, 1),
        exposure_appropriateness=round(exp_score, 1),
        midtone_readability=round(mid_score, 1),
        clipping_health=round(clipping_health_score, 1),
        shadow_saturation_health=round(shadow_sat_score, 1),
        highlight_chroma_health=round(hl_chroma_score, 1),
        source_relative_exposure_change_ev=round(exposure_ev_change, 2),
        hard_gates_passed=hard_gates_passed,
        hard_gate_failures=hard_gate_failures,
        passed=passed,
        diagnosis="; ".join(diag_parts)
    )

def assess_normalization_health(
    shot_id: str,
    source_metrics: ShotMetrics,
    normalized_frames: List[np.ndarray],
    profile: str = "rec709",
    probed_info: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[NormalizationDiagnostics] = None,
    scene_intent: Optional[SceneIntent] = None
) -> NormalizationValidationResult:
    """Evaluates whether the input transform produced a plausible display-referred image
    using profile-aware pre-clip diagnostics, distinguishing introduced clipping from legitimate
    dark/silhouette scenes and expected Log-to-display black mapping.
    """
    if not normalized_frames:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_FAILED",
            passed=False,
            reason="No normalized frames available for verification.",
            metrics_summary={}
        )

    # 1. Finite values check
    all_finite = all(bool(np.all(np.isfinite(f))) for f in normalized_frames)
    if not all_finite:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_FAILED",
            passed=False,
            reason="Non-finite values (NaN or Inf) detected in normalized frames.",
            metrics_summary={"finite_values_passed": False},
            diagnostics=diagnostics
        )

    norm_lums = []
    shadow_clips = []
    highlight_clips = []
    for f in normalized_frames:
        f_bgr = f if f.dtype == np.uint8 else (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2GRAY)
        norm_lums.append(float(np.mean(gray)))
        shadow_clips.append(float(np.mean(gray < 2) * 100.0))
        highlight_clips.append(float(np.mean(gray > 253) * 100.0))
        
    avg_lum = float(np.mean(norm_lums))
    avg_sh_clip = float(np.mean(shadow_clips))
    avg_hl_clip = float(np.mean(highlight_clips))
    
    all_grays = np.concatenate([cv2.cvtColor(f if f.dtype == np.uint8 else (np.clip(f, 0.0, 1.0)*255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).ravel() for f in normalized_frames])
    p5 = float(np.percentile(all_grays, 5))
    p25 = float(np.percentile(all_grays, 25))
    p50 = float(np.percentile(all_grays, 50))
    p75 = float(np.percentile(all_grays, 75))
    p95 = float(np.percentile(all_grays, 95))
    iqr = p75 - p25

    # If diagnostics is not provided, derive pre-clip diagnostics from first frame
    if diagnostics is None and normalized_frames:
        first_f = normalized_frames[0]
        first_float = first_f if first_f.dtype != np.uint8 else (first_f.astype(np.float32) / 255.0)
        _, diagnostics = apply_input_camera_profile_with_diagnostics(
            bgr_float=first_float,
            profile=profile,
            probed_info=probed_info
        )
    
    metrics_summary = {
        "avg_luminance": round(avg_lum, 1),
        "p5": round(p5, 1),
        "p25": round(p25, 1),
        "p50": round(p50, 1),
        "p75": round(p75, 1),
        "p95": round(p95, 1),
        "iqr": round(iqr, 1),
        "shadow_clip_pct": round(avg_sh_clip, 2),
        "highlight_clip_pct": round(avg_hl_clip, 2),
        "positive_luminance_collapsed_to_black_pct": round(diagnostics.positive_luminance_collapsed_to_black_pct, 2) if diagnostics else 0.0,
        "negative_rgb_excursion_pct": round(diagnostics.negative_rgb_excursion_pct, 2) if diagnostics else 0.0
    }

    src_p5 = source_metrics.p5_luminance
    src_iqr = source_metrics.p75_luminance - source_metrics.p25_luminance
    src_transfer = str(probed_info.get("color_transfer", "")).lower() if probed_info else ""

    # Check: Is scene legitimate low-key night or intentional silhouette?
    is_low_key = False
    if scene_intent:
        light_c = scene_intent.lighting_class.lower()
        exp_c = scene_intent.exposure_class.lower()
        if light_c in ["low_key_night", "night", "dark_interior", "intentional_silhouette"] or exp_c in ["low_key", "low_key_underexposed", "intentional_silhouette"]:
            is_low_key = True
    if source_metrics.p50_luminance < 35.0 or source_metrics.avg_luminance < 40.0:
        is_low_key = True

    # 2. Check for double normalization on already display-ready footage
    is_already_rec709 = (
        (src_transfer in ["bt709", "iec61966", "srgb", "smpte170m"] or (src_p5 < 15.0 and src_iqr > 40.0 and source_metrics.avg_chroma > 15.0))
        and profile not in ["rec709", "auto_ask"]
    )
    if is_already_rec709 and (avg_sh_clip > 12.0 or (diagnostics and diagnostics.positive_luminance_collapsed_to_black_pct > 3.0)):
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_WARNING",
            passed=True,
            reason=f"Footage appears already display-ready Rec.709. Applying Log profile '{profile}' has {avg_sh_clip:.1f}% shadow occupancy. Verified with advisory warning.",
            metrics_summary=metrics_summary,
            diagnostics=diagnostics
        )

    # 3. Check for positive-luminance shadow compression
    introduced_clip = diagnostics.positive_luminance_collapsed_to_black_pct if diagnostics else 0.0
    if introduced_clip > 5.0 and not is_low_key:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_WARNING",
            passed=True,
            reason=f"Shadow compression detected: transform mapped {introduced_clip:.1f}% of low-end detail near black floor. Verified with advisory warning.",
            metrics_summary=metrics_summary,
            diagnostics=diagnostics
        )

    # 4. Check for high contrast compression
    if iqr < 8.0 and src_iqr > 20.0:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_WARNING",
            passed=True,
            reason=f"High contrast compression (IQR dropped from {src_iqr:.1f} to {iqr:.1f}). Verified with advisory warning.",
            metrics_summary=metrics_summary,
            diagnostics=diagnostics
        )

    # 5. Check if unexpanded flat Log footage under Rec.709
    if source_metrics.avg_chroma < 12.0 and (src_iqr < 55.0 or source_metrics.p5_luminance > 38.0) and p5 > 25.0:
        if iqr < 35.0 or (p5 > 35.0 and profile in ["rec709", "auto_ask"]):
            return NormalizationValidationResult(
                shot_id=shot_id,
                state="NORMALIZATION_WARNING",
                passed=True,
                reason=f"Footage exhibits elevated black floor (p5={p5:.1f}) and flat contrast (IQR={iqr:.1f}). Verified with advisory warning.",
                metrics_summary=metrics_summary,
                diagnostics=diagnostics
            )

    # 6. Check profile mismatch (e.g. D-Log M vs D-Log)
    if probed_info:
        tr = str(probed_info.get("color_transfer", "")).lower()
        path_str = str(probed_info.get("path", "")).lower()
        if ("dlog_m" in tr or "dlog-m" in tr or "dlog_m" in path_str or "dlog-m" in path_str) and profile == "dji_dlog_dgamut":
            return NormalizationValidationResult(
                shot_id=shot_id,
                state="NORMALIZATION_WARNING",
                passed=True,
                reason="Clip metadata indicates DJI D-Log M under D-Log / D-Gamut selection. Verified with advisory warning.",
                metrics_summary=metrics_summary,
                diagnostics=diagnostics
            )

    # 7. Check clipping after input transform
    if avg_sh_clip > 8.0:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_WARNING",
            passed=True,
            reason=f"Display black occupancy ({avg_sh_clip:.1f}%). Verified with advisory warning.",
            metrics_summary=metrics_summary,
            diagnostics=diagnostics
        )

    # 8. Highlight clipping check
    if avg_hl_clip > 8.0:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_WARNING",
            passed=True,
            reason=f"Elevated highlight occupancy ({avg_hl_clip:.1f}%). Verified with advisory warning.",
            metrics_summary=metrics_summary,
            diagnostics=diagnostics
        )

    # 10. Gamut compression advisory check
    if diagnostics and diagnostics.negative_rgb_excursion_pct > 5.0:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_WARNING",
            passed=True,
            reason=f"Luminance-preserving gamut compression applied to {diagnostics.negative_rgb_excursion_pct:.1f}% of out-of-gamut pixels. Verified with warning.",
            metrics_summary=metrics_summary,
            diagnostics=diagnostics
        )

    return NormalizationValidationResult(
        shot_id=shot_id,
        state="NORMALIZATION_VERIFIED",
        passed=True,
        reason="Normalized display tone distribution healthy and within plausible Rec.709 bounds.",
        metrics_summary=metrics_summary,
        diagnostics=diagnostics
    )

def compute_consistency_score(
    reference: ShotMetrics,
    candidate: ShotMetrics,
    evaluation_mode: str = "same_scene_match",
    ref_plan: Optional[GradePlan] = None,
    cand_plan: Optional[GradePlan] = None,
    scene_intent: Optional[SceneIntent] = None,
    graded_frames: Optional[List[np.ndarray]] = None,
    source_metrics: Optional[ShotMetrics] = None
) -> ConsistencyScore:
    ref_p5 = reference.p5_luminance if reference.p5_luminance > 0 else (reference.sampled_frames[0].p5_luminance if reference.sampled_frames else 0.0)
    ref_p25 = reference.p25_luminance if reference.p25_luminance > 0 else (reference.sampled_frames[0].p25_luminance if reference.sampled_frames else 0.0)
    ref_p50 = reference.p50_luminance if reference.p50_luminance > 0 else reference.avg_luminance
    ref_p75 = reference.p75_luminance if reference.p75_luminance > 0 else (reference.sampled_frames[0].p75_luminance if reference.sampled_frames else 0.0)
    ref_p95 = reference.p95_luminance if reference.p95_luminance > 0 else (reference.sampled_frames[0].p95_luminance if reference.sampled_frames else 255.0)
    
    cand_p5 = candidate.p5_luminance if candidate.p5_luminance > 0 else (candidate.sampled_frames[0].p5_luminance if candidate.sampled_frames else 0.0)
    cand_p25 = candidate.p25_luminance if candidate.p25_luminance > 0 else (candidate.sampled_frames[0].p25_luminance if candidate.sampled_frames else 0.0)
    cand_p50 = candidate.p50_luminance if candidate.p50_luminance > 0 else candidate.avg_luminance
    cand_p75 = candidate.p75_luminance if candidate.p75_luminance > 0 else (candidate.sampled_frames[0].p75_luminance if candidate.sampled_frames else 0.0)
    cand_p95 = candidate.p95_luminance if candidate.p95_luminance > 0 else (candidate.sampled_frames[0].p95_luminance if candidate.sampled_frames else 255.0)

    if evaluation_mode == "same_scene_match":
        # MODE A: Same Scene Technical Match
        # 1. Chromatic Similarity (CIELAB a*, b* centroid distance + chroma consistency)
        l1, a1, b1 = reference.avg_lab_mean
        l2, a2, b2 = candidate.avg_lab_mean
        delta_ab = float(np.sqrt((a1 - a2)**2 + (b1 - b2)**2))
        delta_chroma = abs(reference.avg_chroma - candidate.avg_chroma)
        chroma_score = float(100.0 * np.exp(-(delta_ab + 0.4 * delta_chroma) / 14.0))
        
        # 2. Quantile-based tonal distance
        tonal_err = (
            0.25 * abs(cand_p50 - ref_p50) +
            0.20 * abs(cand_p25 - ref_p25) +
            0.20 * abs(cand_p75 - ref_p75) +
            0.15 * abs(cand_p5 - ref_p5) +
            0.15 * abs(cand_p95 - ref_p95) +
            0.05 * abs(candidate.avg_lab_std[0] - reference.avg_lab_std[0])
        )
        tonal_score = float(100.0 * np.exp(-tonal_err / 20.0))
        
        # 3. Distribution spread distance (interquartile range + L* std)
        cand_iqr = abs(cand_p75 - cand_p25)
        ref_iqr = abs(ref_p75 - ref_p25)
        spread_err = abs(cand_iqr - ref_iqr) + abs(candidate.avg_lab_std[0] - reference.avg_lab_std[0])
        dist_score = float(100.0 * np.exp(-spread_err / 18.0))
        
        # 4. Clipping Health (Advisory only — clipping penalties removed)
        shadow_clip = candidate.avg_shadow_clip_pct if candidate.avg_shadow_clip_pct > 0 else (candidate.sampled_frames[0].shadow_clip_pct if candidate.sampled_frames else 0.0)
        highlight_clip = candidate.avg_highlight_clip_pct if candidate.avg_highlight_clip_pct > 0 else (candidate.sampled_frames[0].highlight_clip_pct if candidate.sampled_frames else 0.0)
        clipping_health = 100.0
        
        base_overall = 0.35 * tonal_score + 0.35 * chroma_score + 0.15 * dist_score + 0.15 * clipping_health
        tonal_gate = min(1.0, 0.25 + 0.75 * (tonal_score / 60.0)) if tonal_score < 60.0 else 1.0
        overall = base_overall * tonal_gate
        
        diagnosis_parts = []
        if tonal_score < 70.0:
            direction = "darker" if cand_p50 < ref_p50 else "brighter"
            diagnosis_parts.append(f"Tonal mismatch: candidate is {direction} than reference (tonal score: {round(tonal_score, 1)})")
        if chroma_score < 70.0:
            diagnosis_parts.append(f"Chromatic cast mismatch (Delta E_ab: {round(delta_ab, 2)})")

        # Check objective scene health if source metrics or frames provided
        if source_metrics is not None or graded_frames is not None:
            eff_src = source_metrics if source_metrics is not None else candidate
            health_score = evaluate_scene_health(
                source_metrics=eff_src,
                graded_metrics=candidate,
                graded_frames=graded_frames,
                scene_intent=scene_intent
            )
            if not health_score.hard_gates_passed:
                overall = float(min(overall, 45.0))
                diagnosis_parts.extend(health_score.hard_gate_failures)
            
        overall = float(np.clip(overall, 0.0, 100.0))
        diagnosis_str = "; ".join(diagnosis_parts) if diagnosis_parts else "Grade harmonized within target tolerance."
        
        return ConsistencyScore(
            overall_score=round(overall, 1),
            tonal_similarity=round(tonal_score, 1),
            chromatic_similarity=round(chroma_score, 1),
            distribution_similarity=round(dist_score, 1),
            clipping_health=round(clipping_health, 1),
            evaluation_mode=evaluation_mode,
            diagnosis=diagnosis_str,
            notes=f"Mode: same_scene_match | Delta E={round(delta_ab, 2)}, Tonal={round(tonal_score, 1)}, ClipHealth={round(clipping_health, 1)}"
        )
    elif evaluation_mode == "reference_baseline":
        eff_src = source_metrics if source_metrics is not None else candidate
        health_score = evaluate_scene_health(
            source_metrics=eff_src,
            graded_metrics=candidate,
            graded_frames=graded_frames,
            scene_intent=scene_intent
        )
        diag = "Master reference baseline standard."
        if not health_score.hard_gates_passed:
            diag += f" Health notice: {'; '.join(health_score.hard_gate_failures)}"
        
        overall = health_score.overall_score if health_score.hard_gates_passed else min(45.0, health_score.overall_score)
        return ConsistencyScore(
            overall_score=round(overall, 1),
            tonal_similarity=100.0,
            chromatic_similarity=100.0,
            distribution_similarity=100.0,
            clipping_health=round(health_score.clipping_health, 1),
            evaluation_mode="reference_baseline",
            diagnosis=diag,
            notes="Mode: reference_baseline | Master technical reference standard."
        )
    else:
        # MODE B: Cross-Scene Look Continuity (Standardized Transform Probes + Image Health)
        if cand_plan is None:
            return ConsistencyScore(
                overall_score=0.0,
                tonal_similarity=0.0,
                chromatic_similarity=0.0,
                distribution_similarity=0.0,
                clipping_health=100.0,
                evaluation_mode=evaluation_mode,
                diagnosis="Ungraded baseline: creative look transform not yet applied.",
                notes="Mode: cross_scene_look_continuity (ungraded baseline)"
            )

        look_score = evaluate_transform_look_continuity(
            ref_plan=ref_plan,
            cand_plan=cand_plan,
            candidate_metrics=candidate
        )
        
        sh_clip = candidate.avg_shadow_clip_pct if candidate.avg_shadow_clip_pct > 0 else 0.0
        hl_clip = candidate.avg_highlight_clip_pct if candidate.avg_highlight_clip_pct > 0 else 0.0
        cand_p5_val = candidate.p5_luminance if candidate.p5_luminance > 0 else 0.0
        cand_p95_val = candidate.p95_luminance if candidate.p95_luminance > 0 else 255.0
        dr = cand_p95_val - cand_p5_val
        
        health_penalty = 0.0
        if dr < 35.0:
            health_penalty += min(20.0, (35.0 - dr) * 1.5)
            
        clipping_health = float(max(40.0, 100.0 - health_penalty))
        
        diag_parts = []
        if look_score.diagnosis and "harmonized" not in look_score.diagnosis.lower():
            diag_parts.append(look_score.diagnosis)

        eff_src = source_metrics if source_metrics is not None else candidate
        health_score = evaluate_scene_health(
            source_metrics=eff_src,
            graded_metrics=candidate,
            graded_frames=graded_frames,
            scene_intent=scene_intent
        )
        
        if not health_score.hard_gates_passed:
            overall = float(min(45.0, look_score.overall_score))
            diag_parts.extend(health_score.hard_gate_failures)
        else:
            overall = 0.35 * look_score.probe_chromatic_harmony + 0.35 * look_score.probe_tonal_continuity + 0.15 * look_score.saturation_scaling_adherence + 0.15 * clipping_health
            if not health_score.passed and health_score.diagnosis and "verified" not in health_score.diagnosis.lower():
                diag_parts.append(health_score.diagnosis)
            
        overall = float(np.clip(overall, 0.0, 100.0))
        diagnosis_str = "; ".join(diag_parts) if diag_parts else "Look invariants harmonized across independent scenes."
        
        return ConsistencyScore(
            overall_score=round(overall, 1),
            tonal_similarity=round(look_score.probe_tonal_continuity, 1),
            chromatic_similarity=round(look_score.probe_chromatic_harmony, 1),
            distribution_similarity=round(look_score.saturation_scaling_adherence, 1),
            clipping_health=round(health_score.clipping_health, 1),
            evaluation_mode=evaluation_mode,
            diagnosis=diagnosis_str,
            notes=f"Mode: cross_scene_look_continuity | LookHarmony={round(look_score.probe_chromatic_harmony, 1)}, ToneContinuity={round(look_score.probe_tonal_continuity, 1)}, Health={round(health_score.overall_score, 1)}"
        )