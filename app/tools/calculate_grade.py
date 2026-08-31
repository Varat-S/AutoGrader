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
    # 1. Base parameters
    contrast = creative_spec.contrast_intent if creative_spec else 1.0
    saturation = creative_spec.saturation_intent if creative_spec else 1.0
    temperature = creative_spec.temperature_shift if creative_spec else 0.0
    tint = creative_spec.tint_shift if creative_spec else 0.0
    
    # 2. Exposure & Black Mist diffusion
    exposure_ev = 0.0
    black_lift = 0.0
    
    if target_semantic:
        exposure_ev = target_semantic.target_exposure_compensation_ev
        black_lift = target_semantic.black_point_lift
        
    if creative_spec and creative_spec.black_mist_diffusion_strength > 0.1:
        # Mist filter lifts shadow density and softens contrast
        mist_strength = creative_spec.black_mist_diffusion_strength
        black_lift += mist_strength * 6.0
        contrast = contrast * (1.0 - mist_strength * 0.08) # Gentle diffusion softening
        
    # 3. Scene Matching vs Scene-Independent Emulation
    if is_reference_shot:
        # Reference shot receives creative film look directly
        return ColorGradeParams(
            exposure_ev=round(exposure_ev, 3),
            contrast=round(contrast, 3),
            pivot=0.5,
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
        
    if is_same_scene:
        # Continuous sequence: full statistical CIELAB matching
        base_params = calculate_deterministic_match_params(reference, target)
        l_gain = base_params.lab_l_gain
        l_offset = base_params.lab_l_offset + black_lift
        a_gain = base_params.lab_a_gain
        a_offset = base_params.lab_a_offset
        b_gain = base_params.lab_b_gain
        b_offset = base_params.lab_b_offset
    else:
        # Independent scenes (different environments):
        # Preserve natural scene luminance, align chromatic white balance and apply film look
        eps = 1e-4
        l_gain = 1.0
        l_offset = round(black_lift, 3)
        
        # Moderate chromatic color balance
        ref_a, ref_b = reference.avg_lab_mean[1], reference.avg_lab_mean[2]
        tgt_a, tgt_b = target.avg_lab_mean[1], target.avg_lab_mean[2]
        
        a_offset = round((ref_a - tgt_a) * 0.35, 3)
        b_offset = round((ref_b - tgt_b) * 0.35, 3)
        a_gain = 1.0
        b_gain = 1.0

    # Skin tone protection
    skin_prot = target_semantic.skin_protection_required if target_semantic else False
    if skin_prot:
        saturation = float(np.clip(saturation, 0.75, 1.20))
        if temperature > 15.0:
            temperature = 15.0

    return ColorGradeParams(
        exposure_ev=round(exposure_ev, 3),
        contrast=round(contrast, 3),
        pivot=0.5,
        saturation=round(saturation, 3),
        temperature=round(temperature, 2),
        tint=round(tint, 2),
        lab_l_gain=round(l_gain, 3),
        lab_l_offset=round(l_offset, 3),
        lab_a_gain=round(a_gain, 3),
        lab_a_offset=round(a_offset, 3),
        lab_b_gain=round(b_gain, 3),
        lab_b_offset=round(b_offset, 3)
    )