from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from app.models.analysis import ShotMetrics, ShotSemanticAnalysis, CreativeSpecification, InputProfileAssessment
from app.models.grade import (
    GradePlan,
    ColorGradeParams,
    InputTransformParams,
    TechnicalBalanceParams,
    SceneMatchParams,
    CreativeLookParams,
    SceneTrimParams,
    OutputTransformParams
)
from app.media.color import calculate_deterministic_match_params, is_log_profile

def parse_highlight_bias_rgb(bias_text: str) -> List[float]:
    # Returns [B, G, R] offsets in normalized [-0.15, +0.15] range
    text = bias_text.lower()
    if "amber" in text or "warm" in text or "golden" in text or "yellow" in text:
        return [-0.04, 0.01, 0.05] # Warm amber (Red up, Blue down)
    elif "cyan" in text or "neon cyan" in text:
        return [0.06, 0.03, -0.05] # Vivid cyan (Blue & Green up, Red down)
    elif "cool" in text or "blue" in text:
        return [0.05, 0.01, -0.04] # Cool blue
    elif "magenta" in text or "pink" in text or "purple" in text:
        return [0.05, -0.03, 0.05]
    elif "green" in text:
        return [-0.02, 0.04, -0.02]
    return [0.0, 0.0, 0.0]

def parse_shadow_bias_rgb(bias_text: str) -> List[float]:
    # Returns [B, G, R] offsets in normalized [-0.15, +0.15] range
    text = bias_text.lower()
    if "slate" in text or "cool" in text or "teal" in text or "blue" in text or "cyan" in text:
        return [0.05, 0.01, -0.03] # Cool slate / teal shadows (Blue up, Red down)
    elif "warm" in text or "brown" in text or "amber" in text:
        return [-0.04, 0.0, 0.04] # Warm shadows
    elif "green" in text:
        return [-0.02, 0.03, -0.02]
    elif "magenta" in text or "purple" in text or "violet" in text:
        return [0.06, -0.03, 0.05] # Deep magenta/purple shadows
    return [0.0, 0.0, 0.0]

def parse_black_level_lift(treatment_text: str, mist_strength: float = 0.0) -> float:
    text = treatment_text.lower()
    lift = 0.0
    if "filmic" in text or "lift" in text or "soft" in text:
        lift += 3.5
    elif "crush" in text or "deep" in text:
        lift -= 2.0
        
    if mist_strength > 0.1:
        lift += mist_strength * 4.0
    return float(np.clip(lift, 0.0, 15.0))

def build_grade_plan(
    reference: ShotMetrics,
    target: ShotMetrics,
    target_semantic: Optional[ShotSemanticAnalysis] = None,
    creative_spec: Optional[CreativeSpecification] = None,
    is_reference_shot: bool = False,
    is_same_scene: bool = False,
    color_profile: str = "auto",
    matched_params: Optional[ColorGradeParams] = None
) -> GradePlan:
    plan = GradePlan(
        shot_id=target.shot_id,
        is_same_scene=is_same_scene
    )
    
    # 1. INPUT TRANSFORM (Authoritative Precedence)
    p_lower = color_profile.lower().strip()
    if p_lower == "auto_ask":
        raise ValueError("auto_ask is a pending decision state and cannot be converted into a GradePlan. A concrete profile must be resolved before grading.")

    if p_lower in ["rec709", "rec.709", "bt709", "srgb", "display"]:
        target_is_log = False
        resolved_profile = "rec709"
    elif "slog3" in p_lower or "s_log3" in p_lower:
        target_is_log = True
        resolved_profile = "sony_slog3_sgamut3cine"
    elif "apple" in p_lower or p_lower == "apple_log_rec2020":
        target_is_log = True
        resolved_profile = "apple_log_rec2020"
    elif p_lower in ["log", "generic log", "generic_log_experimental", "flat"]:
        target_is_log = True
        resolved_profile = "generic_log_experimental"
    else:
        raise ValueError(f"Unsupported camera profile '{color_profile}'. Supported profiles: rec709, sony_slog3_sgamut3cine, apple_log_rec2020, generic_log_experimental.")
        
    plan.input_transform = InputTransformParams(
        is_log=target_is_log,
        profile=resolved_profile,
        log_type="generic_flat",
        black_floor=0.11,
        white_ceil=0.95
    )
    
    # 2. TECHNICAL BALANCE (Primary Exposure EV & White Balance)
    exposure_ev = 0.0
    temp = 0.0
    tint = 0.0
    
    if target_semantic:
        # Positive exposure adjustment = brighten, negative = darken
        rec_ev = getattr(target_semantic, "recommended_exposure_adjustment_ev", None)
        if rec_ev is not None and abs(rec_ev) > 0.001:
            exposure_ev += rec_ev
        else:
            exposure_ev += target_semantic.target_exposure_compensation_ev
            
    if creative_spec:
        temp += creative_spec.temperature_shift
        tint += creative_spec.tint_shift
        
    plan.technical_balance = TechnicalBalanceParams(
        exposure_ev=round(exposure_ev, 3),
        temperature=round(temp, 2),
        tint=round(tint, 2)
    )
    
    # 3. SAME-SCENE SHOT MATCH
    if is_same_scene and not is_reference_shot:
        if matched_params is not None:
            base_match = matched_params
        else:
            base_match = calculate_deterministic_match_params(reference, target, strength=0.90)
            
        plan.scene_match = SceneMatchParams(
            lab_l_gain=base_match.lab_l_gain,
            lab_l_offset=base_match.lab_l_offset,
            lab_a_gain=base_match.lab_a_gain,
            lab_a_offset=base_match.lab_a_offset,
            lab_b_gain=base_match.lab_b_gain,
            lab_b_offset=base_match.lab_b_offset
        )
    else:
        # Independent scene / Reference shot: no forced luminance offset
        plan.scene_match = SceneMatchParams(
            lab_l_gain=1.0,
            lab_l_offset=0.0,
            lab_a_gain=1.0,
            lab_a_offset=0.0,
            lab_b_gain=1.0,
            lab_b_offset=0.0
        )
        
    # 4. SHARED CREATIVE LOOK
    if creative_spec:
        look_contrast = creative_spec.contrast_intent
        look_sat = creative_spec.saturation_intent
        mist = creative_spec.black_mist_diffusion_strength
        
        if getattr(creative_spec, "highlight_rgb_offset", None) and len(creative_spec.highlight_rgb_offset) == 3:
            hl_bias = [float(np.clip(x, -0.15, 0.15)) for x in creative_spec.highlight_rgb_offset]
        else:
            hl_bias = parse_highlight_bias_rgb(creative_spec.highlight_bias)
            
        if getattr(creative_spec, "shadow_rgb_offset", None) and len(creative_spec.shadow_rgb_offset) == 3:
            sh_bias = [float(np.clip(x, -0.15, 0.15)) for x in creative_spec.shadow_rgb_offset]
        else:
            sh_bias = parse_shadow_bias_rgb(creative_spec.shadow_bias)
            
        toe_lift = parse_black_level_lift(creative_spec.black_level_treatment, mist)
        
        plan.creative_look = CreativeLookParams(
            look_title=creative_spec.look_title,
            contrast=round(look_contrast, 3),
            pivot=0.45,
            saturation=round(look_sat, 3),
            shadow_rgb_offset=sh_bias,
            highlight_rgb_offset=hl_bias,
            black_toe_lift=round(toe_lift, 2)
        )
    else:
        # Neutral baseline with zero artificial color bias
        plan.creative_look = CreativeLookParams(
            look_title="Neutral Photographic Baseline",
            contrast=1.0,
            pivot=0.45,
            saturation=1.0,
            shadow_rgb_offset=[0.0, 0.0, 0.0],
            highlight_rgb_offset=[0.0, 0.0, 0.0],
            black_toe_lift=0.0
        )
        
    # 5. SCENE-SPECIFIC TRIM (Preserves natural night/day depth)
    trim_ev = 0.0
    trim_cont = 1.0
    trim_sat = 1.0
    trim_lift = 0.0
    
    if target_semantic:
        if target_semantic.time_of_day == "night" and not is_same_scene:
            # Preserve deep night black floor and mood
            trim_ev -= 0.35
            trim_cont = 1.05
            trim_lift = 1.0
        elif target_semantic.time_of_day == "golden_hour":
            trim_sat = 1.05
            
    plan.scene_trim = SceneTrimParams(
        trim_exposure_ev=round(trim_ev, 3),
        trim_contrast=round(trim_cont, 3),
        trim_saturation=round(trim_sat, 3),
        trim_shadow_lift=round(trim_lift, 2)
    )
    
    # 6. OUTPUT TRANSFORM
    plan.output_transform = OutputTransformParams(
        highlight_shoulder_threshold=0.85,
        highlight_compression_factor=2.0
    )
    
    return plan

def assess_input_profile(
    shot_id: str,
    probed_info: Dict[str, Any],
    metrics: ShotMetrics,
    requested_profile: Optional[str] = None,
    shot_index: int = 0,
    user_confirmed: bool = False
) -> InputProfileAssessment:
    """Calculates an advisory profile assessment, separated into requested,
    recommended, and resolved profiles with explicit confirmation requirements.
    """
    transfer = str(probed_info.get("color_transfer", "")).lower()
    primaries = str(probed_info.get("color_primaries", "")).lower()
    path_lower = str(probed_info.get("path", "")).lower()
    
    metadata_recommendation = None
    if "slog3" in transfer or "s_log3" in transfer or "s-log3" in transfer or "slog3" in path_lower or "s-gamut3" in primaries or "sgamut3" in primaries:
        metadata_recommendation = "sony_slog3_sgamut3cine"
    elif "apple" in transfer or "apple" in path_lower or ("arib-std-b67" in transfer and "bt2020" in primaries):
        metadata_recommendation = "apple_log_rec2020"
    elif "bt709" in transfer or "iec61966" in transfer or "smpte170m" in transfer or "bt709" in primaries:
        metadata_recommendation = "rec709"
        
    p5 = metrics.p5_luminance if metrics.p5_luminance > 0 else (metrics.sampled_frames[0].p5_luminance if metrics.sampled_frames else 0.0)
    p25 = metrics.p25_luminance if metrics.p25_luminance > 0 else (metrics.sampled_frames[0].p25_luminance if metrics.sampled_frames else 0.0)
    p75 = metrics.p75_luminance if metrics.p75_luminance > 0 else (metrics.sampled_frames[0].p75_luminance if metrics.sampled_frames else 0.0)
    p95 = metrics.p95_luminance if metrics.p95_luminance > 0 else (metrics.sampled_frames[0].p95_luminance if metrics.sampled_frames else 255.0)
    iqr = p75 - p25
    chroma = metrics.avg_chroma
    
    reasons = []
    if p5 > 38.0:
        reasons.append(f"elevated black floor (p5={p5:.1f})")
    if iqr < 55.0:
        reasons.append(f"compressed tonal range (IQR={iqr:.1f})")
    if chroma < 12.0:
        reasons.append(f"low baseline chroma ({chroma:.1f})")
    if p95 < 235.0:
        reasons.append(f"compressed highlight ceiling (p95={p95:.1f})")
        
    # Conservative detection (Section 7):
    # Avoid declaring Log from elevated p5 alone (high-key scenes) or low IQR alone (underexposed/dark scenes)
    is_log_suspect = (p5 > 38.0 and chroma < 12.0 and iqr < 55.0 and p95 < 240.0 and metadata_recommendation != "rec709")
    
    if is_log_suspect:
        signal_class_hint = "log_like"
        confidence = 0.85
    elif p5 > 38.0 and chroma < 15.0 and metadata_recommendation != "rec709":
        signal_class_hint = "ambiguous"
        confidence = 0.60
    else:
        signal_class_hint = "display_ready"
        confidence = 0.80
        
    raw_req = (requested_profile or "auto_ask").strip()
    req = raw_req.lower()
    if req in ["apple_log_apple_wide_gamut", "apple_log"]:
        req = "apple_log_rec2020"
    elif req in ["rec.709", "bt709", "display"]:
        req = "rec709"
    elif req in ["slog3", "sony_slog3"]:
        req = "sony_slog3_sgamut3cine"

    warning_msg = None
    requires_confirmation = False
    recommended_profile = None
    resolved_profile = None
    resolution_source = "unresolved"

    # Explicit known profile
    if req in ["rec709", "sony_slog3_sgamut3cine", "apple_log_rec2020", "generic_log_experimental"]:
        resolved_profile = req
        resolution_source = "user_explicit"
        recommended_profile = metadata_recommendation or req
        
        # Check contradictions
        if req == "rec709" and metadata_recommendation in ["sony_slog3_sgamut3cine", "apple_log_rec2020"]:
            requires_confirmation = True
            warning_msg = f"Metadata indicates {metadata_recommendation}, but Rec.709 was selected. Confirm profile selection."
        elif req == "rec709" and signal_class_hint == "log_like":
            requires_confirmation = True
            warning_msg = f"Possible Log/flat footage detected in {shot_id}; select the camera profile if known."
        elif req != "rec709" and signal_class_hint == "display_ready" and p5 < 15.0 and chroma > 16.0:
            requires_confirmation = True
            warning_msg = f"Clip {shot_id} appears display-ready (Rec.709). Applying Log profile '{req}' may crush shadow details or oversaturate the image."
        else:
            requires_confirmation = False

    # Auto-detect requested (auto / auto_ask)
    else:
        if metadata_recommendation:
            recommended_profile = metadata_recommendation
            if req == "auto":
                requires_confirmation = False
                resolved_profile = metadata_recommendation
                resolution_source = "metadata_recommendation"
            else:
                requires_confirmation = True
                resolved_profile = None
                resolution_source = "unresolved"
        else:
            # Advisory detector only - never declare specific profile from histogram alone
            if signal_class_hint == "log_like":
                recommended_profile = "generic_log_experimental"
                requires_confirmation = True
                resolved_profile = None
                resolution_source = "unresolved"
                warning_msg = f"Possible Log/flat footage detected in {shot_id}; select the camera profile if known."
            elif signal_class_hint == "ambiguous":
                recommended_profile = None
                requires_confirmation = True
                resolved_profile = None
                resolution_source = "unresolved"
                warning_msg = f"Ambiguous tonal spread in {shot_id}; select the camera profile if known."
            else:
                recommended_profile = "rec709"
                if req == "auto":
                    requires_confirmation = False
                    resolved_profile = "rec709"
                    resolution_source = "fallback_default"
                else:
                    requires_confirmation = True
                    resolved_profile = None
                    resolution_source = "unresolved"

    if user_confirmed:
        if resolved_profile is None and recommended_profile:
            resolved_profile = recommended_profile
            resolution_source = "user_confirmed"
        requires_confirmation = False

    return InputProfileAssessment(
        shot_index=shot_index,
        shot_id=shot_id,
        requested_profile=raw_req,
        selected_profile=resolved_profile or (recommended_profile if recommended_profile else (raw_req if raw_req not in ["auto", "auto_ask"] else "rec709")),
        metadata_recommendation=metadata_recommendation,
        metadata_hint=metadata_recommendation or "unknown",
        signal_class_hint=signal_class_hint,
        recommended_profile=recommended_profile,
        resolved_profile=resolved_profile,
        resolution_source=resolution_source,
        requires_confirmation=requires_confirmation,
        user_confirmed=user_confirmed,
        warning_message=warning_msg,
        profile_mismatch_warning=bool(requires_confirmation and warning_msg),
        reasons=reasons,
        confidence=round(confidence, 2)
    )