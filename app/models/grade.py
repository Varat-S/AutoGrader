from typing import List, Optional
from pydantic import BaseModel, Field

class ColorGradeParams(BaseModel):
    # Technical adjustments
    exposure_ev: float = Field(0.0, ge=-4.0, le=4.0, description='Exposure adjustment in EV stops')
    contrast: float = Field(1.0, ge=0.5, le=2.0, description='Contrast multiplier (1.0 = neutral)')
    pivot: float = Field(0.5, ge=0.1, le=0.9, description='Contrast pivot in normalized [0, 1]')
    saturation: float = Field(1.0, ge=0.0, le=2.5, description='Saturation multiplier (1.0 = neutral)')
    temperature: float = Field(0.0, ge=-100.0, le=100.0, description='Warmth/coolness adjustment (-100 to +100)')
    tint: float = Field(0.0, ge=-100.0, le=100.0, description='Green/Magenta adjustment (-100 to +100)')
    
    # Statistical Lab transfer parameters
    lab_l_gain: float = Field(1.0, ge=0.2, le=3.0)
    lab_l_offset: float = Field(0.0, ge=-100.0, le=100.0)
    lab_a_gain: float = Field(1.0, ge=0.2, le=3.0)
    lab_a_offset: float = Field(0.0, ge=-100.0, le=100.0)
    lab_b_gain: float = Field(1.0, ge=0.2, le=3.0)
    lab_b_offset: float = Field(0.0, ge=-100.0, le=100.0)
    
    # Lift / Gamma / Gain RGB offsets
    shadow_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    midtone_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    highlight_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

class ConsistencyScore(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0, description='0 to 100 consistency score (higher = closer match)')
    luminance_similarity: float = Field(..., ge=0.0, le=100.0)
    chroma_similarity: float = Field(..., ge=0.0, le=100.0)
    color_distribution_similarity: float = Field(..., ge=0.0, le=100.0)
    notes: Optional[str] = None

class GradeResult(BaseModel):
    reference_shot_id: str
    target_shot_id: str
    params: ColorGradeParams
    lut_path: Optional[str] = None
    output_video_path: Optional[str] = None
    before_consistency: ConsistencyScore
    after_consistency: ConsistencyScore
    explanation: str
