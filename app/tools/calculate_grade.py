from typing import Optional
import numpy as np
from app.models.analysis import ShotMetrics, CreativeSpecification
from app.models.grade import ColorGradeParams
from app.media.color import calculate_deterministic_match_params

def calculate_creative_grade(
    reference: ShotMetrics,
    target: ShotMetrics,
    creative_spec: Optional[CreativeSpecification] = None,
    skin_protection_required: bool = False
) -> ColorGradeParams:
    # 1. Base technical shot match
    base_params = calculate_deterministic_match_params(reference, target)
    
    # 2. Layer creative styling if provided
    if creative_spec:
        # Saturation intent
        sat = base_params.saturation * creative_spec.saturation_intent
        # If skin protection is active, prevent extreme desaturation or oversaturation
        if skin_protection_required:
            sat = float(np.clip(sat, 0.75, 1.25))
            
        # Temperature & Tint from research
        temp = creative_spec.temperature_shift
        tint = creative_spec.tint_shift
        
        # If skin protection is required, moderate warm/magenta shifts
        if skin_protection_required and temp > 15.0:
            temp = 15.0 # Cap warmth so skin does not turn neon orange
            
        contrast = creative_spec.contrast_intent
        
        return ColorGradeParams(
            exposure_ev=base_params.exposure_ev,
            contrast=round(contrast, 3),
            pivot=0.5,
            saturation=round(sat, 3),
            temperature=round(temp, 2),
            tint=round(tint, 2),
            lab_l_gain=base_params.lab_l_gain,
            lab_l_offset=base_params.lab_l_offset,
            lab_a_gain=base_params.lab_a_gain,
            lab_a_offset=base_params.lab_a_offset,
            lab_b_gain=base_params.lab_b_gain,
            lab_b_offset=base_params.lab_b_offset
        )
        
    return base_params