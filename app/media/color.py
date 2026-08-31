import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from app.models.analysis import FrameMetrics, ShotMetrics
from app.models.grade import ColorGradeParams, ConsistencyScore

def compute_frame_metrics(bgr_frame: np.ndarray, timestamp_sec: float = 0.0) -> FrameMetrics:
    if bgr_frame.dtype != np.uint8:
        bgr_frame = np.clip(bgr_frame * 255.0, 0, 255).astype(np.uint8)
    
    h, w, _ = bgr_frame.shape
    total_pixels = h * w
    
    b, g, r = bgr_frame[:, :, 0], bgr_frame[:, :, 1], bgr_frame[:, :, 2]
    r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    
    # Rec.709 Luminance
    y = 0.2126 * r.astype(np.float32) + 0.7152 * g.astype(np.float32) + 0.0722 * b.astype(np.float32)
    mean_lum = float(np.mean(y))
    median_lum = float(np.median(y))
    p5_lum = float(np.percentile(y, 5))
    p95_lum = float(np.percentile(y, 95))
    
    shadow_clip = float(np.sum(y < 2.0) / total_pixels * 100.0)
    highlight_clip = float(np.sum(y > 253.0) / total_pixels * 100.0)
    
    # Perceptual CIELAB
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
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
        avg_lab_mean=[round(avg_l, 2), round(avg_a, 2), round(avg_b, 2)],
        avg_lab_std=[round(avg_l_std, 2), round(avg_a_std, 2), round(avg_b_std, 2)],
        avg_chroma=round(avg_chroma, 2),
        dominant_cast=cast
    )

def is_log_profile(metrics: ShotMetrics) -> bool:
    p5_avg = float(np.mean([f.p5_luminance for f in metrics.sampled_frames]))
    return (p5_avg > 38.0 and metrics.avg_chroma < 12.0)

def calculate_deterministic_match_params(
    reference: ShotMetrics,
    target: ShotMetrics,
    strength: float = 1.0
) -> ColorGradeParams:
    ref_l_mean, ref_a_mean, ref_b_mean = reference.avg_lab_mean
    ref_l_std, ref_a_std, ref_b_std = reference.avg_lab_std
    
    tgt_l_mean, tgt_a_mean, tgt_b_mean = target.avg_lab_mean
    tgt_l_std, tgt_a_std, tgt_b_std = target.avg_lab_std
    
    eps = 1e-4
    l_gain = float(np.clip(ref_l_std / (tgt_l_std + eps), 0.7, 1.6))
    a_gain = float(np.clip(ref_a_std / (tgt_a_std + eps), 0.6, 1.8))
    b_gain = float(np.clip(ref_b_std / (tgt_b_std + eps), 0.6, 1.8))
    
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
        pivot=0.5,
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

def apply_log_to_rec709_cst(bgr_float: np.ndarray) -> np.ndarray:
    # 1. Expand flat black floor (D-Log / S-Log3 code value ~30 / 255)
    black_floor = 0.11
    white_ceil = 0.95
    img = np.clip((bgr_float - black_floor) / (white_ceil - black_floor), 0.0, 1.0)
    
    # 2. Sigmoidal film S-curve for rich contrast
    p = 0.40
    c = 1.35
    below = p * (np.maximum(0.0, img / p) ** c)
    above = 1.0 - (1.0 - p) * (np.maximum(0.0, (1.0 - img) / (1.0 - p)) ** c)
    img = np.where(img < p, below, above)
    
    # 3. Gamut expansion to Rec.709
    uint8_img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.65, 0.0, 255.0)
    rec709_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    return np.clip(rec709_bgr, 0.0, 1.0)

def apply_color_grade_to_frame(bgr_frame: np.ndarray, params: ColorGradeParams, is_log: bool = False) -> np.ndarray:
    # 1. Floating-point working representation [0, 1]
    img = np.clip(bgr_frame.astype(np.float32) / 255.0, 0.0, 1.0)
    
    # 2. Apply Log -> Rec.709 Color Space Transform if input is Log
    if is_log:
        img = apply_log_to_rec709_cst(img)
    
    # 3. Exposure
    if abs(params.exposure_ev) > 0.01:
        img = img * float(2.0 ** params.exposure_ev)
        
    # 4. Filmic Contrast & Tone Curve around pivot
    if abs(params.contrast - 1.0) > 0.01:
        x_norm = np.clip(img, 0.0, 1.0)
        p = float(np.clip(params.pivot, 0.05, 0.95))
        c = float(params.contrast)
        below = p * (np.maximum(0.0, x_norm / p) ** c)
        above = 1.0 - (1.0 - p) * (np.maximum(0.0, (1.0 - x_norm) / (1.0 - p)) ** c)
        img = np.where(x_norm < p, below, above)
        
    # 5. White balance (Temperature / Tint)
    if abs(params.temperature) > 0.1 or abs(params.tint) > 0.1:
        temp_f = params.temperature / 150.0
        tint_f = params.tint / 150.0
        img[:, :, 2] = img[:, :, 2] * (1.0 + temp_f) # Red
        img[:, :, 0] = img[:, :, 0] * (1.0 - temp_f) # Blue
        img[:, :, 1] = img[:, :, 1] * (1.0 - tint_f) # Green

    # 6. Filmic Highlight Protection Shoulder (prevents hard clipping in highlights & foam)
    img = np.where(img > 0.85, 0.85 + (img - 0.85) / (1.0 + (img - 0.85) * 2.0), img)
    img = np.clip(img, 0.0, 1.0)
    
    # 7. CIELAB Perceptual Color Alignment
    uint8_img = (img * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    l = lab[:, :, 0] * (100.0 / 255.0)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    
    # Mild shadow lift only if explicitly requested
    if abs(params.lab_l_offset) > 0.01:
        shadow_lift_weight = np.clip((55.0 - l) / 55.0, 0.0, 1.0) ** 2.0
        effective_l_offset = params.lab_l_offset * shadow_lift_weight
    else:
        effective_l_offset = 0.0
    
    l_graded = np.clip(params.lab_l_gain * l + effective_l_offset, 0.0, 100.0)
    a_graded = np.clip(params.lab_a_gain * a + params.lab_a_offset, -127.0, 127.0)
    b_graded = np.clip(params.lab_b_gain * b + params.lab_b_offset, -127.0, 127.0)
    
    lab_out = np.zeros_like(lab)
    lab_out[:, :, 0] = l_graded * (255.0 / 100.0)
    lab_out[:, :, 1] = a_graded + 128.0
    lab_out[:, :, 2] = b_graded + 128.0
    
    bgr_graded = cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    
    # 8. Saturation in HSV space
    if abs(params.saturation - 1.0) > 0.01:
        hsv = cv2.cvtColor((np.clip(bgr_graded, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.saturation, 0.0, 255.0)
        bgr_graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
    return np.clip(bgr_graded * 255.0, 0, 255).astype(np.uint8)

def compute_consistency_score(shot_a: ShotMetrics, shot_b: ShotMetrics) -> ConsistencyScore:
    # 1. Color Palette & Chromatic Coherence (Delta E in chromatic a*, b* planes)
    a1, b1 = shot_a.avg_lab_mean[1], shot_a.avg_lab_mean[2]
    a2, b2 = shot_b.avg_lab_mean[1], shot_b.avg_lab_mean[2]
    delta_chroma_e = float(np.sqrt((a1 - a2)**2 + (b1 - b2)**2))
    chroma_score = float(100.0 * np.exp(-delta_chroma_e / 18.0))
    
    # 2. Dynamic Range & Tonal Depth Health (rewarding proper black floor & highlight headroom)
    # Healthy black floor: p5 between 8 and 35
    p5_b = float(np.mean([f.p5_luminance for f in shot_b.sampled_frames]))
    if p5_b > 45.0: # washed out / flat
        tonal_score = max(40.0, 100.0 - (p5_b - 35.0) * 2.0)
    elif p5_b < 2.0: # crushed
        tonal_score = 80.0
    else:
        tonal_score = 95.0
        
    # 3. Overall Coherence
    overall = 0.60 * chroma_score + 0.40 * tonal_score
    overall = float(np.clip(overall, 0.0, 100.0))
    
    return ConsistencyScore(
        overall_score=round(overall, 1),
        luminance_similarity=round(tonal_score, 1),
        chroma_similarity=round(chroma_score, 1),
        color_distribution_similarity=round(chroma_score, 1),
        notes=f"Chromatic Delta E={round(delta_chroma_e, 2)}, Tonal Depth={round(tonal_score, 1)}"
    )