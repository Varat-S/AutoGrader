from typing import Optional
import numpy as np
from app.models.analysis import ShotMetrics, ShotSemanticAnalysis, CreativeSpecification
from app.models.grade import ColorGradeParams
from app.media.color import calculate_deterministic_match_params

def calculate_creative_grade(
    reference: ShotMetrics,
    target: ShotMetrics,
    target_semantic: Optional[ShotSemanticAnalysis] = None,
    creative_spec: Optional[CreativeSpecification] = None,
    is_reference_shot: bool = False,
    is_same_scene: bool = False
) -> ColorGradeParams:
    # 1. Base creative parameters
    contrast = creative_spec.contrast_intent if creative_spec else 1.0
    saturation = creative_spec.saturation_intent if creative_spec else 1.0
    temperature = creative_spec.temperature_shift if creative_spec else 0.0
    tint = creative_spec.tint_shift if creative_spec else 0.0
    
    # 2. Exposure & Black Point Lift
    exposure_ev = 0.0
    black_lift = 0.0
    
    if target_semantic:
        exposure_ev = target_semantic.target_exposure_compensation_ev
        black_lift = target_semantic.black_point_lift
        
    if creative_spec and creative_spec.black_mist_diffusion_strength > 0.1:
        mist_strength = creative_spec.black_mist_diffusion_strength
        black_lift += mist_strength * 3.5
        contrast = contrast * (1.0 - mist_strength * 0.04)
        
    # 3. Reference Shot vs Target Shot
    if is_reference_shot:
        return ColorGradeParams(
            exposure_ev=round(exposure_ev, 3),
            contrast=round(contrast, 3),
            pivot=0.45,
            saturation=round(saturation, 3),
            temperature=round(temperature, 2),
            tint=round(tint, 2),
            lab_l_gain=1.0,
            lab_l_offset=round(black_lift, 3),
            lab_a_gain=1.0,
            lab_a_offset=0.0,
            lab_b_gain=1.0,
            lab_b_offset=0.0
        )
        
    # For target shots:
    # Chromatic balance alignment (a*, b* channels) without forcing luminance
    base_params = calculate_deterministic_match_params(reference, target, strength=0.7)
    a_gain = base_params.lab_a_gain
    a_offset = base_params.lab_a_offset
    b_gain = base_params.lab_b_gain
    b_offset = base_params.lab_b_offset
    
    # Skin protection dampening
    if target_semantic and target_semantic.skin_protection_required:
        a_offset *= 0.5
        b_offset *= 0.5
        
    return ColorGradeParams(
        exposure_ev=round(exposure_ev, 3),
        contrast=round(contrast, 3),
        pivot=0.45,
        saturation=round(saturation, 3),
        temperature=round(temperature, 2),
        tint=round(tint, 2),
        lab_l_gain=1.0, # Independent luminance preservation
        lab_l_offset=round(black_lift, 3),
        lab_a_gain=round(a_gain, 3),
        lab_a_offset=round(a_offset, 3),
        lab_b_gain=round(b_gain, 3),
        lab_b_offset=round(b_offset, 3)
    )