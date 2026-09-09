from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class InputProfile(str, Enum):
    REC709 = "rec709"
    SONY_SLOG3 = "sony_slog3_sgamut3cine"
    APPLE_LOG = "apple_log_rec2020"
    GENERIC_LOG = "generic_log_experimental"
    AUTO_ASK = "auto_ask"

class ShotProfileSelection(BaseModel):
    shot_index: int = Field(..., ge=0, description="0-indexed shot position in sequence")
    profile: InputProfile = Field(InputProfile.REC709, description="Selected camera input profile")
    user_confirmed: bool = Field(False, description="Whether user explicitly confirmed the profile")
    override_warning: bool = Field(False, description="Whether user explicitly overrides any warnings")

    @field_validator("profile", mode="before")
    @classmethod
    def normalize_profile(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ["apple_log_apple_wide_gamut", "apple_log"]:
                return InputProfile.APPLE_LOG
            if v_clean in ["slog3", "sony_slog3"]:
                return InputProfile.SONY_SLOG3
            if v_clean in ["rec.709", "bt709", "display"]:
                return InputProfile.REC709
            if v_clean in ["generic_log", "generic log", "flat"]:
                return InputProfile.GENERIC_LOG
        return v

class InputTransformParams(BaseModel):
    is_log: bool = Field(False, description="Whether input is flat / logarithmic profile requiring normalization")
    profile: str = Field("rec709", description="Camera profile: rec709, sony_slog3_sgamut3cine, apple_log_rec2020, generic_log_experimental")
    log_type: str = Field("generic_flat", description="Legacy alias for backwards compatibility")
    black_floor: float = Field(0.11, description="Normalized sensor black point")
    white_ceil: float = Field(0.95, description="Normalized sensor clipping ceiling")

    @field_validator("profile")
    @classmethod
    def validate_concrete_profile(cls, v: str) -> str:
        if v.strip().lower() == "auto_ask":
            raise ValueError("auto_ask is a pending decision state and cannot be used as an input transform profile in a GradePlan.")
        return v

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

from app.models.analysis import LookContinuityScore, SceneHealthScore

class EffectiveGradeSummary(BaseModel):
    shot_id: str
    scene_group_id: str = "group_1"
    scene_class: str = "daylight"
    scene_rationale: str = ""
    input_profile: Dict[str, Any] = Field(default_factory=dict)
    technical_exposure_ev: float = 0.0
    scene_match_exposure_ev: float = 0.0
    scene_trim_exposure_ev: float = 0.0
    total_exposure_ev: float = 0.0
    shared_contrast: float = 1.0
    scene_contrast_trim: float = 1.0
    effective_contrast: float = 1.0
    shared_saturation: float = 1.0
    scene_saturation_trim: float = 1.0
    effective_saturation: float = 1.0
    revision_state: str = "ACCEPTED"

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

    def compute_effective_summary(
        self,
        scene_group_id: str = "group_1",
        scene_class: str = "daylight",
        scene_rationale: str = "",
        input_profile_dict: Optional[Dict[str, Any]] = None,
        camera_profile: Optional[str] = None,
        revision_state: str = "ACCEPTED"
    ) -> EffectiveGradeSummary:
        total_exposure = round(self.technical_balance.exposure_ev + self.scene_trim.trim_exposure_ev, 3)
        effective_contrast = round(self.creative_look.contrast * self.scene_trim.trim_contrast, 4)
        effective_sat = round(self.creative_look.saturation * self.scene_trim.trim_saturation, 4)

        input_dict = input_profile_dict or {"resolved": camera_profile or self.input_transform.profile}
        if "resolved" not in input_dict:
            input_dict["resolved"] = camera_profile or self.input_transform.profile

        return EffectiveGradeSummary(
            shot_id=self.shot_id,
            scene_group_id=scene_group_id,
            scene_class=scene_class,
            scene_rationale=scene_rationale,
            input_profile=input_dict,
            technical_exposure_ev=round(self.technical_balance.exposure_ev, 3),
            scene_match_exposure_ev=0.0,
            scene_trim_exposure_ev=round(self.scene_trim.trim_exposure_ev, 3),
            total_exposure_ev=total_exposure,
            shared_contrast=round(self.creative_look.contrast, 3),
            scene_contrast_trim=round(self.scene_trim.trim_contrast, 3),
            effective_contrast=effective_contrast,
            shared_saturation=round(self.creative_look.saturation, 3),
            scene_saturation_trim=round(self.scene_trim.trim_saturation, 3),
            effective_saturation=effective_sat,
            revision_state=revision_state
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
    evaluation_mode: str = Field("same_scene_match", description="same_scene_match or cross_scene_look_continuity")
    grade_summary: Optional[EffectiveGradeSummary] = None
    look_continuity: Optional[LookContinuityScore] = None
    scene_health: Optional[SceneHealthScore] = None
    before_proxy_path: Optional[str] = None
    after_proxy_path: Optional[str] = None
    original_source_path: Optional[str] = None