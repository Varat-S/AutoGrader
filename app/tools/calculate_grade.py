from typing import Optional, List, Dict, Any
import numpy as np
from app.models.analysis import ShotMetrics, ShotSemanticAnalysis, CreativeSpecification
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
    elif "cyan" in text or "cool" in text or "blue" in text:
        return [0.04, 0.01, -0.04] # Cool cyan
    elif "magenta" in text or "pink" in text:
        return [0.03, -0.02, 0.04]
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
    elif "magenta" in text:
        return [0.03, -0.02, 0.03]
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
    if color_profile == "Rec.709":
        target_is_log = False
    elif color_profile in ["Log", "Generic Log"]:
        target_is_log = True
    else:
        # Auto: check metadata + heuristic + semantic text
        target_is_log = bool(is_log_profile(target) or (target_semantic and "log" in target_semantic.scene_description.lower()))
        
    plan.input_transform = InputTransformParams(
        is_log=target_is_log,
        log_type="generic_flat",
        black_floor=0.11,
        white_ceil=0.95
    )
    
    # 2. TECHNICAL BALANCE (Primary Exposure EV & White Balance)
    exposure_ev = 0.0
    temp = 0.0
    tint = 0.0
    
    if target_semantic:
        exposure_ev += target_semantic.target_exposure_compensation_ev
        
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
        
        hl_bias = parse_highlight_bias_rgb(creative_spec.highlight_bias)
        sh_bias = parse_shadow_bias_rgb(creative_spec.shadow_bias)
        toe_lift = parse_black_level_lift(creative_spec.black_level_treatment, mist)
        
        plan.creative_look = CreativeLookParams(
            contrast=round(look_contrast, 3),
            pivot=0.45,
            saturation=round(look_sat, 3),
            shadow_rgb_offset=sh_bias,
            highlight_rgb_offset=hl_bias,
            black_toe_lift=round(toe_lift, 2)
        )
    else:
        plan.creative_look = CreativeLookParams(
            contrast=1.10,
            pivot=0.45,
            saturation=1.05,
            shadow_rgb_offset=[0.05, 0.01, -0.03], # cool slate
            highlight_rgb_offset=[-0.04, 0.01, 0.05], # warm amber
            black_toe_lift=3.0
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