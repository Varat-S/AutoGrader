import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
from app.models.analysis import FrameMetrics, ShotMetrics, NormalizationValidationResult
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

def is_log_profile(metrics: ShotMetrics) -> bool:
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

# 2. Apple Log / Apple Wide Gamut (Apple Log Profile White Paper 2023)
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

def apply_input_camera_profile(
    bgr_float: np.ndarray,
    profile: str = "rec709",
    black_floor: float = 0.11,
    white_ceil: float = 0.95
) -> np.ndarray:
    p = profile.lower().strip()
    if p in ["rec709", "bt709", "srgb", "display"]:
        return np.clip(bgr_float, 0.0, 1.0)
        
    if "slog3" in p or "s_log3" in p:
        rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
        linear_rgb = sony_slog3_to_linear(rgb)
        h, w, c_dim = linear_rgb.shape
        reshaped = linear_rgb.reshape(-1, 3)
        converted = np.dot(reshaped, MAT_SGAMUT3CINE_TO_BT709.T).reshape(h, w, c_dim)
        display_rgb = scene_linear_to_rec709_display(converted)
        return cv2.cvtColor(display_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
        
    if "apple_log" in p or "apple" in p:
        rgb = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2RGB)
        linear_rgb = apple_log_to_linear(rgb)
        h, w, c_dim = linear_rgb.shape
        reshaped = linear_rgb.reshape(-1, 3)
        converted = np.dot(reshaped, MAT_BT2020_TO_BT709.T).reshape(h, w, c_dim)
        display_rgb = scene_linear_to_rec709_display(converted)
        return cv2.cvtColor(display_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
        
    # Default to generic experimental flat CST
    return apply_log_to_rec709_cst(bgr_float, black_floor=black_floor, white_ceil=white_ceil)

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
        
    l_offset = float(ref_l_mean - l_gain * tgt_l_mean)
    a_offset = float(ref_a_mean - a_gain * tgt_a_mean)
    b_offset = float(ref_b_mean - b_gain * tgt_b_mean)
    
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
        
        shoulder_thresh = 0.85
        compression_factor = 2.0

    # STAGE 1: Input Transform / Camera Profile Normalization
    profile = "rec709"
    if isinstance(plan_or_params, GradePlan):
        profile = plan_or_params.input_transform.profile
        if profile in ["rec709", "auto_ask"] and (plan_or_params.input_transform.is_log or is_log):
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

    # D. Saturation in Pure Float32 HSV space
    if abs(saturation - 1.0) > 0.005:
        hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0.0, 1.0)
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
    candidate_metrics: ShotMetrics
) -> Tuple[float, float, float, float, List[str]]:
    """Evaluates cross-scene look continuity by testing creative-only plans on standardized probes.
    Excludes input transforms, per-shot technical balance, scene matching, and scene trims.
    """
    diagnosis = []
    if ref_plan is None or cand_plan is None:
        return 95.0, 95.0, 95.0, 95.0, ["Reference baseline established."]
        
    # Isolate creative look ONLY (excluding input transform, technical balance, scene match, scene trim)
    clean_ref = GradePlan(shot_id="clean_ref", creative_look=ref_plan.creative_look)
    clean_cand = GradePlan(shot_id="clean_cand", creative_look=cand_plan.creative_look)
    
    ramp, patches = create_creative_probe_set()
    
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
        
    # 4. Candidate Image Health (independent of reference luminance)
    sh_clip = candidate_metrics.avg_shadow_clip_pct
    hl_clip = candidate_metrics.avg_highlight_clip_pct
    cand_p5 = candidate_metrics.p5_luminance if candidate_metrics.p5_luminance > 0 else 0.0
    cand_p95 = candidate_metrics.p95_luminance if candidate_metrics.p95_luminance > 0 else 255.0
    dr = cand_p95 - cand_p5
    
    health_penalty = 0.0
    if sh_clip > 2.0:
        health_penalty += min(25.0, (sh_clip - 2.0) * 5.0)
    if hl_clip > 2.0:
        health_penalty += min(25.0, (hl_clip - 2.0) * 5.0)
    if dr < 35.0:
        health_penalty += min(20.0, (35.0 - dr) * 1.5)
        
    clipping_health = float(max(40.0, 100.0 - health_penalty))
    if clipping_health < 80.0:
        diagnosis.append(f"Excessive candidate clipping (shadow: {sh_clip:.1f}%, highlight: {hl_clip:.1f}%)")
        
    # Tonal continuity combines contrast curve adherence and highlight split adherence
    tonal_continuity = 0.60 * contrast_score + 0.40 * hl_score
    # Chromatic harmony combines saturation scaling and shadow split adherence
    chromatic_harmony = 0.60 * sat_score + 0.40 * sh_score
    
    return float(tonal_continuity), float(chromatic_harmony), float(sat_score), float(clipping_health), diagnosis

def assess_normalization_health(
    shot_id: str,
    source_metrics: ShotMetrics,
    normalized_frames: List[np.ndarray],
    profile: str = "rec709"
) -> NormalizationValidationResult:
    """Evaluates whether the input transform produced a plausible display-referred image."""
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
    p75 = float(np.percentile(all_grays, 75))
    p95 = float(np.percentile(all_grays, 95))
    iqr = p75 - p25
    
    metrics_summary = {
        "avg_luminance": round(avg_lum, 1),
        "p5": round(p5, 1),
        "p95": round(p95, 1),
        "iqr": round(iqr, 1),
        "shadow_clip_pct": round(avg_sh_clip, 2),
        "highlight_clip_pct": round(avg_hl_clip, 2)
    }
    
    # If source had flat log-like characteristics:
    src_iqr = source_metrics.p75_luminance - source_metrics.p25_luminance
    if source_metrics.avg_chroma < 12.0 and (src_iqr < 55.0 or source_metrics.p5_luminance > 38.0):
        # Clip was flat initially. If it didn't materially expand or p5 is still elevated under Rec.709:
        if iqr < 35.0 or (p5 > 35.0 and profile in ["rec709", "auto_ask"]):
            return NormalizationValidationResult(
                shot_id=shot_id,
                state="PROFILE_CONFIRMATION_REQUIRED",
                passed=False,
                reason=f"Footage exhibits elevated black floor (p5={p5:.1f}) and flat contrast (IQR={iqr:.1f}). Verification of camera Log profile required.",
                metrics_summary=metrics_summary
            )
            
    if avg_sh_clip > 8.0 or avg_hl_clip > 8.0:
        return NormalizationValidationResult(
            shot_id=shot_id,
            state="NORMALIZATION_FAILED",
            passed=False,
            reason=f"Excessive clipping after input transform (Shadow: {avg_sh_clip:.1f}%, Highlight: {avg_hl_clip:.1f}%).",
            metrics_summary=metrics_summary
        )
        
    return NormalizationValidationResult(
        shot_id=shot_id,
        state="NORMALIZATION_VERIFIED",
        passed=True,
        reason="Normalized display tone distribution healthy and within plausible Rec.709 bounds.",
        metrics_summary=metrics_summary
    )

def compute_consistency_score(
    reference: ShotMetrics,
    candidate: ShotMetrics,
    evaluation_mode: str = "same_scene_match",
    ref_plan: Optional[GradePlan] = None,
    cand_plan: Optional[GradePlan] = None
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
        
        # 4. Clipping Health
        shadow_clip = candidate.avg_shadow_clip_pct if candidate.avg_shadow_clip_pct > 0 else (candidate.sampled_frames[0].shadow_clip_pct if candidate.sampled_frames else 0.0)
        highlight_clip = candidate.avg_highlight_clip_pct if candidate.avg_highlight_clip_pct > 0 else (candidate.sampled_frames[0].highlight_clip_pct if candidate.sampled_frames else 0.0)
        clip_penalty = min(70.0, shadow_clip * 3.0 + highlight_clip * 4.0)
        clipping_health = float(max(0.0, 100.0 - clip_penalty))
        
        base_overall = 0.35 * tonal_score + 0.35 * chroma_score + 0.15 * dist_score + 0.15 * clipping_health
        tonal_gate = min(1.0, 0.25 + 0.75 * (tonal_score / 60.0)) if tonal_score < 60.0 else 1.0
        overall = base_overall * tonal_gate
        
        diagnosis_parts = []
        if tonal_score < 70.0:
            direction = "darker" if cand_p50 < ref_p50 else "brighter"
            diagnosis_parts.append(f"Tonal mismatch: candidate is {direction} than reference (tonal score: {round(tonal_score, 1)})")
        if chroma_score < 70.0:
            diagnosis_parts.append(f"Chromatic cast mismatch (Delta E_ab: {round(delta_ab, 2)})")
        if clipping_health < 80.0:
            diagnosis_parts.append(f"Excessive clipping (shadow: {round(shadow_clip, 1)}%, highlight: {round(highlight_clip, 1)}%)")
            
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
    else:
        # MODE B: Cross-Scene Look Continuity (Standardized Transform Probes + Image Health)
        tonal_score, chroma_score, sat_score, clipping_health, diag_parts = evaluate_transform_look_continuity(
            ref_plan=ref_plan,
            cand_plan=cand_plan,
            candidate_metrics=candidate
        )
        overall = 0.35 * chroma_score + 0.35 * tonal_score + 0.15 * sat_score + 0.15 * clipping_health
        overall = float(np.clip(overall, 0.0, 100.0))
        diagnosis_str = "; ".join(diag_parts) if diag_parts else "Look invariants harmonized across independent scenes."
        
        return ConsistencyScore(
            overall_score=round(overall, 1),
            tonal_similarity=round(tonal_score, 1),
            chromatic_similarity=round(chroma_score, 1),
            distribution_similarity=round(sat_score, 1),
            clipping_health=round(clipping_health, 1),
            evaluation_mode=evaluation_mode,
            diagnosis=diagnosis_str,
            notes=f"Mode: cross_scene_look_continuity | LookHarmony={round(chroma_score, 1)}, ToneContinuity={round(tonal_score, 1)}, ClipHealth={round(clipping_health, 1)}"
        )