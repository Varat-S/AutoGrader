import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
from app.models.analysis import FrameMetrics, ShotMetrics
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
    
    # Perceptual CIELAB
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
    avg_lum = float(np.mean([m.mean_luminance for m in frame_metrics]))
    p5_avg = float(np.mean([m.p5_luminance for m in frame_metrics]))
    p25_avg = float(np.mean([m.p25_luminance for m in frame_metrics]))
    p50_avg = float(np.mean([m.p50_luminance for m in frame_metrics]))
    p75_avg = float(np.mean([m.p75_luminance for m in frame_metrics]))
    p95_avg = float(np.mean([m.p95_luminance for m in frame_metrics]))
    
    shadow_clip_avg = float(np.mean([m.shadow_clip_pct for m in frame_metrics]))
    highlight_clip_avg = float(np.mean([m.highlight_clip_pct for m in frame_metrics]))
    
    avg_l = float(np.mean([m.lab_l_mean for m in frame_metrics]))
    avg_a = float(np.mean([m.lab_a_mean for m in frame_metrics]))
    avg_b = float(np.mean([m.lab_b_mean for m in frame_metrics]))
    
    avg_l_std = float(np.mean([m.lab_l_std for m in frame_metrics]))
    avg_a_std = float(np.mean([m.lab_a_std for m in frame_metrics]))
    avg_b_std = float(np.mean([m.lab_b_std for m in frame_metrics]))
    avg_chroma = float(np.mean([m.mean_chroma for m in frame_metrics]))
    
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
    return (p5_val > 38.0 and metrics.avg_chroma < 12.0)

def apply_log_to_rec709_cst(bgr_float: np.ndarray, black_floor: float = 0.11, white_ceil: float = 0.95) -> np.ndarray:
    # 1. Expand flat black floor
    img = np.clip((bgr_float - black_floor) / max(0.1, white_ceil - black_floor), 0.0, 1.0)
    
    # 2. Sigmoidal S-curve mapping for Log-to-Rec.709 expansion
    p = 0.40
    c = 1.35
    below = p * (np.maximum(0.0, img / p) ** c)
    above = 1.0 - (1.0 - p) * (np.maximum(0.0, (1.0 - img) / (1.0 - p)) ** c)
    img = np.where(img < p, below, above)
    
    # 3. Wide Gamut expansion to Rec.709
    uint8_img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.65, 0.0, 255.0)
    rec709_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    return np.clip(rec709_bgr, 0.0, 1.0)

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
    l_offset = l_offset * strength
    a_offset = a_offset * strength
    b_offset = b_offset * strength
    
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
    # 1. 32-bit Floating-point representation in [0.0, 1.0]
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

    # STAGE 1: Input Transform / Log Normalization
    if do_log:
        img = apply_log_to_rec709_cst(img, black_floor=log_floor, white_ceil=log_ceil)
        
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
    
    # STAGE 3: Same-Scene Shot Match (CIELAB Perceptual Alignment)
    if (abs(l_gain - 1.0) > 0.001 or abs(l_offset) > 0.001 or
        abs(a_gain - 1.0) > 0.001 or abs(a_offset) > 0.001 or
        abs(b_gain - 1.0) > 0.001 or abs(b_offset) > 0.001):
        
        uint8_img = (img * 255.0).astype(np.uint8)
        lab = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        l = lab[:, :, 0] * (100.0 / 255.0)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        
        l_graded = np.clip(l_gain * l + l_offset, 0.0, 100.0)
        a_graded = np.clip(a_gain * a + a_offset, -127.0, 127.0)
        b_graded = np.clip(b_gain * b + b_offset, -127.0, 127.0)
        
        lab_out = np.zeros_like(lab)
        lab_out[:, :, 0] = l_graded * (255.0 / 100.0)
        lab_out[:, :, 1] = a_graded + 128.0
        lab_out[:, :, 2] = b_graded + 128.0
        
        img = cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    # STAGE 4: Shared Creative Look (Filmic Contrast, Highlight/Shadow Tints, Saturation)
    # A. Filmic S-Curve Contrast
    if abs(contrast - 1.0) > 0.005:
        p = float(np.clip(pivot, 0.05, 0.95))
        c = float(contrast)
        x_norm = np.clip(img, 0.0, 1.0)
        below = p * (np.maximum(0.0, x_norm / p) ** c)
        above = 1.0 - (1.0 - p) * (np.maximum(0.0, (1.0 - x_norm) / (1.0 - p)) ** c)
        img = np.where(x_norm < p, below, above)

    # B. Luminance-Weighted Highlight and Shadow Biases (Split Toning)
    # Luma map in [0.0, 1.0]
    lum = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    
    # Highlight weight (smooth ramp for lum > 0.55)
    w_highlight = np.clip((lum - 0.55) / 0.40, 0.0, 1.0) ** 1.5
    # Shadow weight (smooth ramp for lum < 0.45)
    w_shadow = np.clip((0.45 - lum) / 0.40, 0.0, 1.0) ** 1.5
    
    if highlight_bias and any(abs(x) > 0.001 for x in highlight_bias):
        # highlight_bias format: [B, G, R]
        img[:, :, 0] += highlight_bias[0] * w_highlight
        img[:, :, 1] += highlight_bias[1] * w_highlight
        img[:, :, 2] += highlight_bias[2] * w_highlight
        
    if shadow_bias and any(abs(x) > 0.001 for x in shadow_bias):
        # shadow_bias format: [B, G, R]
        img[:, :, 0] += shadow_bias[0] * w_shadow
        img[:, :, 1] += shadow_bias[1] * w_shadow
        img[:, :, 2] += shadow_bias[2] * w_shadow
        
    # C. Black Toe Density / Lift
    if abs(black_toe_lift) > 0.01:
        toe_f = black_toe_lift / 255.0
        toe_weight = np.clip((0.40 - lum) / 0.40, 0.0, 1.0) ** 2.0
        img += toe_f * np.expand_dims(toe_weight, axis=-1)

    img = np.clip(img, 0.0, 1.0)

    # D. Saturation in HSV space
    if abs(saturation - 1.0) > 0.005:
        uint8_img = (img * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0.0, 255.0)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

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

def compute_consistency_score(
    reference: ShotMetrics,
    candidate: ShotMetrics,
    evaluation_mode: str = "same_scene_match"
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

    # 1. Chromatic Similarity (CIELAB a*, b* centroid distance + chroma consistency)
    l1, a1, b1 = reference.avg_lab_mean
    l2, a2, b2 = candidate.avg_lab_mean
    delta_ab = float(np.sqrt((a1 - a2)**2 + (b1 - b2)**2))
    delta_chroma = abs(reference.avg_chroma - candidate.avg_chroma)
    chroma_score = float(100.0 * np.exp(-(delta_ab + 0.4 * delta_chroma) / 14.0))
    
    # 2. Clipping Health Penalty
    shadow_clip = candidate.avg_shadow_clip_pct if candidate.avg_shadow_clip_pct > 0 else (candidate.sampled_frames[0].shadow_clip_pct if candidate.sampled_frames else 0.0)
    highlight_clip = candidate.avg_highlight_clip_pct if candidate.avg_highlight_clip_pct > 0 else (candidate.sampled_frames[0].highlight_clip_pct if candidate.sampled_frames else 0.0)
    clip_penalty = min(70.0, shadow_clip * 3.0 + highlight_clip * 4.0)
    clipping_health = float(max(0.0, 100.0 - clip_penalty))

    diagnosis_parts = []
    
    if evaluation_mode == "same_scene_match":
        # MODE A: Same Scene Technical Match
        # Quantile-based tonal distance
        tonal_err = (
            0.25 * abs(cand_p50 - ref_p50) +
            0.20 * abs(cand_p25 - ref_p25) +
            0.20 * abs(cand_p75 - ref_p75) +
            0.15 * abs(cand_p5 - ref_p5) +
            0.15 * abs(cand_p95 - ref_p95) +
            0.05 * abs(candidate.avg_lab_std[0] - reference.avg_lab_std[0])
        )
        tonal_score = float(100.0 * np.exp(-tonal_err / 20.0))
        
        # Distribution spread distance (interquartile range + L* std)
        cand_iqr = abs(cand_p75 - cand_p25)
        ref_iqr = abs(ref_p75 - ref_p25)
        spread_err = abs(cand_iqr - ref_iqr) + abs(candidate.avg_lab_std[0] - reference.avg_lab_std[0])
        dist_score = float(100.0 * np.exp(-spread_err / 18.0))
        
        base_overall = 0.35 * tonal_score + 0.35 * chroma_score + 0.15 * dist_score + 0.15 * clipping_health
        # Severe tonal mismatch bottlenecks same-scene match
        tonal_gate = min(1.0, 0.25 + 0.75 * (tonal_score / 60.0)) if tonal_score < 60.0 else 1.0
        overall = base_overall * tonal_gate
        
        if tonal_score < 70.0:
            direction = "darker" if cand_p50 < ref_p50 else "brighter"
            diagnosis_parts.append(f"Tonal mismatch: candidate is {direction} than reference (tonal score: {round(tonal_score, 1)})")
        if chroma_score < 70.0:
            diagnosis_parts.append(f"Chromatic cast mismatch (Delta E_ab: {round(delta_ab, 2)})")
        if clipping_health < 80.0:
            diagnosis_parts.append(f"Excessive clipping (shadow: {round(shadow_clip, 1)}%, highlight: {round(highlight_clip, 1)}%)")
            
    else:
        # MODE B: Cross-Scene Look Continuity (Day vs Night / Different Lighting)
        # Evaluates look invariants without penalizing darker night scene baseline
        cand_range = cand_p95 - cand_p5
        toe_health = 100.0 if (cand_p5 <= 35.0 and cand_p5 >= 2.0) else max(40.0, 100.0 - abs(cand_p5 - 18.0) * 2.5)
        range_health = min(100.0, (cand_range / 45.0) * 100.0)
        headroom_health = 100.0 if cand_p95 <= 248.0 else max(20.0, 100.0 - (cand_p95 - 248.0) * 10.0)
        tonal_score = float(0.40 * toe_health + 0.40 * range_health + 0.20 * headroom_health)
        
        # 2. Saturation Discipline
        sat_health = 100.0 if (candidate.avg_chroma >= 6.0 and candidate.avg_chroma <= 32.0) else max(40.0, 100.0 - abs(candidate.avg_chroma - 16.0) * 3.0)
        dist_score = float(sat_health)
        
        overall = 0.40 * chroma_score + 0.30 * tonal_score + 0.15 * dist_score + 0.15 * clipping_health
        
        if tonal_score < 70.0:
            diagnosis_parts.append(f"Tonal depth flaw in scene (toe/contrast score: {round(tonal_score, 1)})")
        if chroma_score < 70.0:
            diagnosis_parts.append(f"Cross-scene palette deviation (Delta E_ab: {round(delta_ab, 2)})")
        if clipping_health < 80.0:
            diagnosis_parts.append(f"Clipping detected (shadow: {round(shadow_clip, 1)}%, highlight: {round(highlight_clip, 1)}%)")

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
        notes=f"Mode: {evaluation_mode} | Delta E={round(delta_ab, 2)}, Tonal={round(tonal_score, 1)}, ClipHealth={round(clipping_health, 1)}"
    )