from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class InputProfile(str, Enum):
    REC709 = "rec709"
    SONY_SLOG3 = "sony_slog3_sgamut3cine"
    APPLE_LOG = "apple_log_apple_wide_gamut"
    GENERIC_LOG = "generic_log_experimental"
    AUTO_ASK = "auto_ask"

class ShotProfileSelection(BaseModel):
    shot_index: int = Field(..., ge=0, description="0-indexed shot position in sequence")
    profile: InputProfile = Field(InputProfile.REC709, description="Selected camera input profile")
    user_confirmed: bool = Field(False, description="Whether user explicitly confirmed the profile")

class InputTransformParams(BaseModel):
    is_log: bool = Field(False, description="Whether input is flat / logarithmic profile requiring normalization")
    profile: str = Field("rec709", description="Camera profile: rec709, sony_slog3_sgamut3cine, apple_log_apple_wide_gamut, generic_log_experimental, auto_ask")
    log_type: str = Field("generic_flat", description="Legacy alias for backwards compatibility")
    black_floor: float = Field(0.11, description="Normalized sensor black point")
    white_ceil: float = Field(0.95, description="Normalized sensor clipping ceiling")

class TechnicalBalanceParams(BaseModel):
    exposure_ev: float = Field(0.0, ge=-4.0, le=4.0, description="Per-shot primary exposure balance in EV stops (positive = brighten, negative = darken)")
    temperature: float = Field(0.0, ge=-100.0, le=100.0, description="White balance temperature correction (-100 to +100)")
    tint: float = Field(0.0, ge=-100.0, le=100.0, description="White balance green/magenta tint correction (-100 to +100)")

class SceneMatchParams(BaseModel):
    lab_l_gain: float = Field(1.0, ge=0.5, le=2.0, description="Same-scene tonal contrast alignment")
    lab_l_offset: float = Field(0.0, ge=-100.0, le=100.0, description="Same-scene luminance offset (0.0 for independent scenes)")
    lab_a_gain: float = Field(1.0, ge=0.4, le=2.5, description="Same-scene green-red chromatic gain")
    lab_a_offset: float = Field(0.0, ge=-100.0, le=100.0, description="Same-scene green-red chromatic shift")
    lab_b_gain: float = Field(1.0, ge=0.4, le=2.5, description="Same-scene blue-yellow chromatic gain")
    lab_b_offset: float = Field(0.0, ge=-100.0, le=100.0, description="Same-scene blue-yellow chromatic shift")

class CreativeLookParams(BaseModel):
    look_title: str = Field("Default Creative Look", description="Title of the creative film emulation")
    contrast: float = Field(1.0, ge=0.5, le=2.0, description="Filmic contrast multiplier")
    pivot: float = Field(0.45, ge=0.1, le=0.9, description="Contrast S-curve midtone pivot")
    saturation: float = Field(1.0, ge=0.0, le=2.5, description="Creative color saturation multiplier")
    shadow_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Luminance-weighted shadow tint [B, G, R]")
    highlight_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Luminance-weighted highlight tint [B, G, R]")
    black_toe_lift: float = Field(0.0, ge=0.0, le=30.0, description="Filmic shadow toe density / lift")
    black_mist_strength: float = Field(0.0, ge=0.0, le=1.0, description="Optical Black Mist diffusion emulation intensity")

class SceneTrimParams(BaseModel):
    trim_exposure_ev: float = Field(0.0, ge=-2.0, le=2.0, description="Scene-specific mood / day-night trim")
    trim_contrast: float = Field(1.0, ge=0.7, le=1.4, description="Scene-specific contrast trim")
    trim_saturation: float = Field(1.0, ge=0.7, le=1.4, description="Scene-specific saturation trim")
    trim_shadow_lift: float = Field(0.0, ge=-20.0, le=20.0, description="Scene-specific shadow toe trim")

class OutputTransformParams(BaseModel):
    highlight_shoulder_threshold: float = Field(0.85, ge=0.7, le=0.95, description="Luminance threshold where compressive shoulder starts")
    highlight_compression_factor: float = Field(2.0, ge=1.0, le=4.0, description="Soft roll-off compression curve slope")
    clip_protection: bool = Field(True, description="Enforce digital clipping protection")

class GradePlan(BaseModel):
    shot_id: str
    is_same_scene: bool = Field(False, description="True if target is in the same lighting context as reference")
    input_transform: InputTransformParams = Field(default_factory=InputTransformParams)
    technical_balance: TechnicalBalanceParams = Field(default_factory=TechnicalBalanceParams)
    scene_match: SceneMatchParams = Field(default_factory=SceneMatchParams)
    creative_look: CreativeLookParams = Field(default_factory=CreativeLookParams)
    scene_trim: SceneTrimParams = Field(default_factory=SceneTrimParams)
    output_transform: OutputTransformParams = Field(default_factory=OutputTransformParams)

    def to_legacy_params(self) -> "ColorGradeParams":
        # Combines staged parameters into unified ColorGradeParams for backwards compatibility
        total_exposure = self.technical_balance.exposure_ev + self.scene_trim.trim_exposure_ev
        total_contrast = self.creative_look.contrast * self.scene_trim.trim_contrast
        total_saturation = self.creative_look.saturation * self.scene_trim.trim_saturation
        
        return ColorGradeParams(
            exposure_ev=round(total_exposure, 3),
            contrast=round(total_contrast, 3),
            pivot=self.creative_look.pivot,
            saturation=round(total_saturation, 3),
            temperature=self.technical_balance.temperature,
            tint=self.technical_balance.tint,
            lab_l_gain=self.scene_match.lab_l_gain,
            lab_l_offset=self.scene_match.lab_l_offset + self.creative_look.black_toe_lift,
            lab_a_gain=self.scene_match.lab_a_gain,
            lab_a_offset=self.scene_match.lab_a_offset,
            lab_b_gain=self.scene_match.lab_b_gain,
            lab_b_offset=self.scene_match.lab_b_offset,
            shadow_rgb_offset=self.creative_look.shadow_rgb_offset,
            highlight_rgb_offset=self.creative_look.highlight_rgb_offset
        )

class ColorGradeParams(BaseModel):
    exposure_ev: float = Field(0.0, ge=-4.0, le=4.0)
    contrast: float = Field(1.0, ge=0.5, le=2.0)
    pivot: float = Field(0.45, ge=0.1, le=0.9)
    saturation: float = Field(1.0, ge=0.0, le=2.5)
    temperature: float = Field(0.0, ge=-100.0, le=100.0)
    tint: float = Field(0.0, ge=-100.0, le=100.0)
    
    lab_l_gain: float = Field(1.0, ge=0.2, le=3.0)
    lab_l_offset: float = Field(0.0, ge=-100.0, le=100.0)
    lab_a_gain: float = Field(1.0, ge=0.2, le=3.0)
    lab_a_offset: float = Field(0.0, ge=-100.0, le=100.0)
    lab_b_gain: float = Field(1.0, ge=0.2, le=3.0)
    lab_b_offset: float = Field(0.0, ge=-100.0, le=100.0)
    
    shadow_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    midtone_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    highlight_rgb_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

class ConsistencyScore(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall consistency score (0-100)")
    tonal_similarity: float = Field(..., ge=0.0, le=100.0, description="Quantile-based tonal/luminance similarity or probe tone response")
    chromatic_similarity: float = Field(..., ge=0.0, le=100.0, description="CIELAB centroid similarity or probe split-tone harmony")
    distribution_similarity: float = Field(..., ge=0.0, le=100.0, description="Tonal/chromatic spread or saturation scaling adherence")
    clipping_health: float = Field(..., ge=0.0, le=100.0, description="Penalty for shadow crush (<2) or highlight blow-out (>253)")
    evaluation_mode: str = Field("same_scene_match", description="same_scene_match or cross_scene_look_continuity")
    diagnosis: Optional[str] = Field(None, description="Diagnostic feedback for autonomous revision")
    notes: Optional[str] = None

class RevisionRecord(BaseModel):
    iteration: int
    state: str = Field(..., description="INITIAL_EVALUATION, ACCEPTED, REVISION_PROPOSED, REVISION_IMPROVED, REVISION_REJECTED, NO_ACTIONABLE_REVISION, MAX_REVISIONS_REACHED")
    action_taken: str
    overall_score_before: float
    overall_score_after: Optional[float] = None
    diagnosis: str
    parameter_deltas: Dict[str, Any] = Field(default_factory=dict)

class GradeResult(BaseModel):
    reference_shot_id: str
    target_shot_id: str
    state: str = Field("ACCEPTED", description="Final revision state machine outcome")
    plan: Optional[GradePlan] = None
    params: ColorGradeParams
    lut_path: Optional[str] = None
    shared_lut_path: Optional[str] = None
    output_video_path: Optional[str] = None
    before_consistency: ConsistencyScore
    after_consistency: ConsistencyScore
    revisions_performed: int = 0
    history: List[RevisionRecord] = Field(default_factory=list)
    explanation: str