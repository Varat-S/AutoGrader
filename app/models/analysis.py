import math
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator

class ShotProfileDecision(BaseModel):
    shot_index: int = Field(0, ge=0)
    shot_id: str
    requested_profile: str = "rec709"
    metadata_recommendation: Optional[str] = None
    signal_class_hint: Literal["display_ready", "log_like", "ambiguous"] = "display_ready"
    recommended_profile: Optional[str] = None
    resolved_profile: Optional[str] = None
    resolution_source: Literal["user_explicit", "metadata_recommendation", "user_confirmed", "user_override", "fallback_default", "unresolved"] = "unresolved"
    requires_confirmation: bool = False
    user_confirmed: bool = False
    warning_message: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    confidence: float = 0.5

class InputProfileAssessment(ShotProfileDecision):
    selected_profile: str = "rec709"
    metadata_hint: str = "unknown"
    profile_mismatch_warning: bool = False

class NormalizationDiagnostics(BaseModel):
    requested_profile: str = "rec709"
    resolved_profile: str = "rec709"
    source_color_range: str = "tv"
    source_transfer_metadata: str = "unknown"
    expected_log_black_code: float = 0.0
    decoded_nonpositive_luminance_pct: float = 0.0
    positive_luminance_collapsed_to_black_pct: float = 0.0
    negative_rgb_excursion_pct: float = 0.0
    over_one_rgb_excursion_pct: float = 0.0
    post_black_occupancy_pct: float = 0.0
    post_highlight_occupancy_pct: float = 0.0
    source_p5: float = 0.0
    source_p25: float = 0.0
    source_p50: float = 0.0
    source_p75: float = 0.0
    source_p95: float = 0.0
    normalized_p5: float = 0.0
    normalized_p25: float = 0.0
    normalized_p50: float = 0.0
    normalized_p75: float = 0.0
    normalized_p95: float = 0.0
    source_iqr: float = 0.0
    normalized_iqr: float = 0.0
    finite_values_passed: bool = True

class NormalizationValidationResult(BaseModel):
    shot_id: str
    state: Literal[
        "NORMALIZATION_VERIFIED",
        "NORMALIZATION_WARNING",
        "PROFILE_CONFIRMATION_REQUIRED",
        "NORMALIZATION_WARNING_OVERRIDDEN",
        "NORMALIZATION_FAILED"
    ]
    passed: bool
    reason: str
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Optional[NormalizationDiagnostics] = None

class FrameMetrics(BaseModel):
    timestamp_sec: float
    mean_luminance: float = Field(..., description="Mean luminance (0-255)")
    median_luminance: float = Field(..., description="Median luminance (0-255)")
    p5_luminance: float = Field(..., description="5th percentile luminance (black point indicator)")
    p25_luminance: float = Field(0.0, description="25th percentile luminance (shadow tone indicator)")
    p50_luminance: float = Field(0.0, description="50th percentile luminance (midtone indicator)")
    p75_luminance: float = Field(0.0, description="75th percentile luminance (upper midtone indicator)")
    p95_luminance: float = Field(..., description="95th percentile luminance (highlight indicator)")
    shadow_clip_pct: float = Field(..., description="Percentage of pixels clipped in shadows (<2/255)")
    highlight_clip_pct: float = Field(..., description="Percentage of pixels clipped in highlights (>253/255)")
    lab_l_mean: float
    lab_l_std: float
    lab_a_mean: float
    lab_a_std: float
    lab_b_mean: float
    lab_b_std: float
    mean_chroma: float
    r_mean: float
    g_mean: float
    b_mean: float

class ShotMetrics(BaseModel):
    shot_id: str
    video_path: str
    duration_sec: float
    width: int
    height: int
    fps: float
    sampled_frames: List[FrameMetrics]
    avg_luminance: float
    p5_luminance: float = 0.0
    p25_luminance: float = 0.0
    p50_luminance: float = 0.0
    p75_luminance: float = 0.0
    p95_luminance: float = 0.0
    avg_shadow_clip_pct: float = 0.0
    avg_highlight_clip_pct: float = 0.0
    avg_lab_mean: List[float] = Field(..., description="[L, a, b] average mean")
    avg_lab_std: List[float] = Field(..., description="[L, a, b] average std")
    avg_chroma: float
    dominant_cast: str = "neutral"

class ShotSemanticAnalysis(BaseModel):
    shot_id: str
    scene_group_id: str = Field("group_1", description="Logical scene group identifier e.g. group_1, group_2")
    relationship_to_reference: str = Field("same_scene", description="Relationship: reference, same_scene, independent_scene")
    scene_description: str = Field(..., description="Brief description of visual content and setting")
    lighting_environment: str = Field(..., description="e.g. outdoor daylight, golden hour, overcast, indoor tungsten, mixed")
    time_of_day: str = Field(..., description="e.g. day, night, golden_hour, dusk, dawn")
    exposure_assessment: str = Field("balanced", description="e.g. balanced, underexposed, overexposed, high_key, low_key")
    recommended_exposure_adjustment_ev: float = Field(0.0, ge=-4.0, le=4.0, description="Recommended exposure adjustment in EV stops (positive = brighten, negative = darken)")
    target_exposure_compensation_ev: float = Field(0.0, ge=-4.0, le=4.0, description="Legacy alias for recommended_exposure_adjustment_ev (positive = brighten, negative = darken)")
    black_point_lift: float = Field(0.0, ge=0.0, le=20.0, description="Shadow toe lift to emulate soft filmic shadow density")
    people_present: bool = Field(False, description="True if human subjects are present in frame")
    dominant_color_cast: str = Field(..., description="Visual perception of color temperature or tint")
    reference_suitability_score: float = Field(..., ge=0.0, le=1.0, description="Suitability score (0-1) to serve as technical color reference")
    intentional_light_sources: List[str] = Field(default_factory=list, description="Practical lights that should intentionally remain warm/cool")
    likely_neutral_objects: List[str] = Field(default_factory=list, description="Objects in frame likely to be neutral white/gray")
    key_composition_elements: List[str] = Field(default_factory=list, description="Dominant visual anchors in scene")

class SequenceInspectionResult(BaseModel):
    shots: List[ShotSemanticAnalysis]
    recommended_reference_shot_id: str = Field(..., description="ID of the optimal technical reference shot")
    scene_relationship: str = Field("mixed_sequence", description="continuous_sequence, independent_scenes, or mixed_sequence")

class SearchCitation(BaseModel):
    title: str
    url: str
    excerpt: str
    extracted_principle: Optional[str] = None
    influence: Optional[str] = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v_clean = str(v).strip()
        if not (v_clean.startswith("http://") or v_clean.startswith("https://")):
            raise ValueError(f"Citation URL must start with http:// or https://, got '{v}'")
        return v_clean

class CinematographyResearchResult(BaseModel):
    query: str
    objective: str
    sources: List[SearchCitation]
    synthesized_principles: List[str] = Field(default_factory=list, description="Core cinematography principles extracted from sources")
    is_grounded: bool = Field(True, description="True if Parallel returned genuine verified research sources")

class GlobalLookIntent(BaseModel):
    look_title: str = Field("Default Creative Look", description="Title of the shared film emulation")
    palette_relationship: str = Field("harmonious", description="Palette harmony: harmonious, complementary, monochromatic, discordant")
    base_contrast: float = Field(1.0, ge=0.5, le=1.8, description="Sequence-wide contrast multiplier")
    base_saturation: float = Field(1.0, ge=0.0, le=2.0, description="Sequence-wide saturation multiplier")
    shadow_bias: str = Field("neutral", description="Shadow split tint intent")
    highlight_bias: str = Field("neutral", description="Highlight split tint intent")
    highlight_rgb_offset: Optional[List[float]] = Field(default=None, description="Direct [B, G, R] offset in [-0.15, 0.15] for highlights")
    shadow_rgb_offset: Optional[List[float]] = Field(default=None, description="Direct [B, G, R] offset in [-0.15, 0.15] for shadows")
    black_level_character: str = Field("neutral", description="Black point character: neutral, filmic lifted, deep crushed")
    highlight_rolloff: str = Field("filmic soft", description="Highlight roll-off characteristic")
    global_temperature_intent: float = Field(0.0, ge=-50.0, le=50.0, description="Global temperature bias")
    global_tint_intent: float = Field(0.0, ge=-50.0, le=50.0, description="Global tint bias")
    black_mist_diffusion_strength: float = Field(0.0, ge=0.0, le=1.0, description="Black Mist diffusion emulation intensity")

class SceneIntent(BaseModel):
    scene_group_id: str = Field(..., description="Scene group identifier matching ShotSemanticAnalysis.scene_group_id")
    lighting_class: str = Field("daylight", description="daylight, golden_hour, low_key_night, practical_night, interior_tungsten, overcast, high_key, intentional_silhouette")
    exposure_class: str = Field("balanced", description="balanced, low_key, high_key, underexposed, overexposed, low_key_underexposed, intentional_silhouette, overexposed_night")
    scene_mood: str = Field("natural", description="Artistic mood of this specific scene")
    source_relative_exposure_bounds: List[float] = Field(default_factory=lambda: [-1.0, 1.0], description="Allowed [min_ev, max_ev] delta relative to source")
    target_tonal_rules: List[str] = Field(default_factory=list, description="Target tonal rules e.g. preserve deep shadows, maintain readable face midtones")
    
    # Distinct separation: multiplier bounds vs measured health ceilings
    scene_saturation_trim_bounds: List[float] = Field(default_factory=lambda: [0.70, 1.20], description="Allowed deterministic multiplier range [min_sat_trim, max_sat_trim]")
    shadow_output_chroma_ceiling: float = Field(0.35, ge=0.05, le=1.0, description="Measured output-image shadow chroma ceiling in normalized HSV [0, 1] space")
    midtone_output_chroma_ceiling: float = Field(0.70, ge=0.10, le=1.0, description="Measured output-image midtone chroma ceiling in normalized HSV [0, 1] space")

    # Legacy field aliases maintained for backwards-compatibility
    shadow_saturation_ceiling: float = Field(0.85, ge=0.05, le=2.0, description="Legacy alias for shadow saturation threshold")
    midtone_saturation_ceiling: float = Field(1.30, ge=0.10, le=2.5, description="Legacy alias for midtone saturation threshold")
    contrast_trim_bounds: List[float] = Field(default_factory=lambda: [0.85, 1.15], description="Allowed [min_contrast_trim, max_contrast_trim]")
    saturation_trim_bounds: List[float] = Field(default_factory=lambda: [0.70, 1.20], description="Allowed [min_sat_trim, max_sat_trim]")
    protected_visual_anchors: List[str] = Field(default_factory=list, description="Visual anchors that must be protected from clipping or tint corruption")
    confidence: float = Field(0.90, ge=0.0, le=1.0, description="Confidence in scene classification")
    concise_rationale: str = Field("", description="Concise rationale for scene exposure and trim intent")

    @field_validator("source_relative_exposure_bounds")
    @classmethod
    def validate_exp_bounds(cls, v: Any) -> List[float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"source_relative_exposure_bounds must be a 2-element list [min_ev, max_ev], got {v}")
        min_ev, max_ev = float(v[0]), float(v[1])
        if not (math.isfinite(min_ev) and math.isfinite(max_ev)):
            raise ValueError("Exposure bounds must contain finite numbers (no NaN or Inf)")
        if min_ev >= max_ev:
            raise ValueError(f"Reversed or empty bounds: min_ev ({min_ev}) must be strictly less than max_ev ({max_ev})")
        if min_ev < -2.5 or max_ev > 2.5:
            raise ValueError(f"Exposure bounds [{min_ev}, {max_ev}] exceed safe application limits [-2.5, 2.5]")
        return [round(min_ev, 3), round(max_ev, 3)]

    @field_validator("contrast_trim_bounds")
    @classmethod
    def validate_contrast_bounds(cls, v: Any) -> List[float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"contrast_trim_bounds must be a 2-element list [min, max], got {v}")
        min_c, max_c = float(v[0]), float(v[1])
        if not (math.isfinite(min_c) and math.isfinite(max_c)):
            raise ValueError("Contrast bounds must contain finite numbers (no NaN or Inf)")
        if min_c >= max_c:
            raise ValueError(f"Reversed or empty bounds: min ({min_c}) must be strictly less than max ({max_c})")
        if min_c < 0.5 or max_c > 1.8:
            raise ValueError(f"Contrast bounds [{min_c}, {max_c}] exceed safe application limits [0.5, 1.8]")
        return [round(min_c, 3), round(max_c, 3)]

    @field_validator("saturation_trim_bounds", "scene_saturation_trim_bounds")
    @classmethod
    def validate_sat_bounds(cls, v: Any) -> List[float]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"saturation_trim_bounds must be a 2-element list [min, max], got {v}")
        min_s, max_s = float(v[0]), float(v[1])
        if not (math.isfinite(min_s) and math.isfinite(max_s)):
            raise ValueError("Saturation bounds must contain finite numbers (no NaN or Inf)")
        if min_s >= max_s:
            raise ValueError(f"Reversed or empty bounds: min ({min_s}) must be strictly less than max ({max_s})")
        if min_s < 0.4 or max_s > 2.0:
            raise ValueError(f"Saturation bounds [{min_s}, {max_s}] exceed safe application limits [0.4, 2.0]")
        return [round(min_s, 3), round(max_s, 3)]

    @property
    def rationale(self) -> str:
        return self.concise_rationale

class LookContinuityScore(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall look continuity score (0-100)")
    contrast_slope_adherence: float = Field(100.0, ge=0.0, le=100.0)
    highlight_split_adherence: float = Field(100.0, ge=0.0, le=100.0)
    shadow_split_adherence: float = Field(100.0, ge=0.0, le=100.0)
    saturation_scaling_adherence: float = Field(100.0, ge=0.0, le=100.0)
    probe_tonal_continuity: float = Field(100.0, ge=0.0, le=100.0)
    probe_chromatic_harmony: float = Field(100.0, ge=0.0, le=100.0)
    passed: bool = Field(True, description="True if passes look continuity threshold (>= 75)")
    diagnosis: Optional[str] = None

    def __iter__(self):
        return iter((
            self.probe_tonal_continuity,
            self.probe_chromatic_harmony,
            self.saturation_scaling_adherence,
            100.0,
            [self.diagnosis] if self.diagnosis else []
        ))

class SceneHealthScore(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall scene health score (0-100)")
    exposure_appropriateness: float = Field(100.0, ge=0.0, le=100.0)
    midtone_readability: float = Field(100.0, ge=0.0, le=100.0)
    clipping_health: float = Field(100.0, ge=0.0, le=100.0)
    shadow_saturation_health: float = Field(100.0, ge=0.0, le=100.0)
    highlight_chroma_health: float = Field(100.0, ge=0.0, le=100.0)
    source_relative_exposure_change_ev: float = 0.0
    hard_gates_passed: bool = Field(True, description="False if any critical health gate failed")
    hard_gate_failures: List[str] = Field(default_factory=list, description="Descriptions of any failed hard gates")
    passed: bool = Field(True, description="True if passed both overall score and hard gates")
    diagnosis: Optional[str] = None

class CreativeSpecification(BaseModel):
    look_title: str
    target_aesthetic: str
    synthesis_mode: Literal["grounded", "ungrounded", "fallback"] = Field("ungrounded", description="Provenance of the creative specification")
    contrast_intent: float = Field(1.0, ge=0.5, le=1.8, description="Target contrast multiplier")
    saturation_intent: float = Field(1.0, ge=0.0, le=2.0, description="Target saturation multiplier")
    highlight_bias: str = Field("neutral", description="e.g. warm amber, neutral, soft golden, neon cyan")
    shadow_bias: str = Field("neutral", description="e.g. cool teal, deep blue, magenta, neutral")
    highlight_rgb_offset: Optional[List[float]] = Field(default=None, description="Direct [B, G, R] offset in [-0.15, 0.15] for highlights")
    shadow_rgb_offset: Optional[List[float]] = Field(default=None, description="Direct [B, G, R] offset in [-0.15, 0.15] for shadows")
    black_level_treatment: str = Field("neutral", description="e.g. filmic lifted, deep crushed, neutral")
    temperature_shift: float = Field(0.0, ge=-50.0, le=50.0, description="Creative temperature bias")
    tint_shift: float = Field(0.0, ge=-50.0, le=50.0, description="Creative tint bias")
    black_mist_diffusion_strength: float = Field(0.0, ge=0.0, le=1.0, description="Black Mist diffusion emulation intensity")
    fallback_reason: Optional[str] = Field(default=None, description="Detailed error reason if fell back to neutral baseline")
    cinematography_principles: List[str] = Field(default_factory=list)
    citations: List[SearchCitation] = Field(default_factory=list)
    global_look: Optional[GlobalLookIntent] = Field(default=None, description="Structured sequence-wide global look intent")
    scene_intents: List[SceneIntent] = Field(default_factory=list, description="Per-scene-group intent list")

    def get_canonical_global_look(self) -> GlobalLookIntent:
        if self.global_look is not None:
            return self.global_look
        look = GlobalLookIntent(
            look_title=self.look_title,
            base_contrast=self.contrast_intent,
            base_saturation=self.saturation_intent,
            shadow_bias=self.shadow_bias,
            highlight_bias=self.highlight_bias,
            highlight_rgb_offset=self.highlight_rgb_offset,
            shadow_rgb_offset=self.shadow_rgb_offset,
            black_level_character=self.black_level_treatment,
            global_temperature_intent=self.temperature_shift,
            global_tint_intent=self.tint_shift,
            black_mist_diffusion_strength=self.black_mist_diffusion_strength
        )
        self.global_look = look
        return look

    def normalize_canonical_look(self) -> "CreativeSpecification":
        canonical = self.get_canonical_global_look()
        self.look_title = canonical.look_title
        self.contrast_intent = canonical.base_contrast
        self.saturation_intent = canonical.base_saturation
        self.highlight_bias = canonical.highlight_bias
        self.shadow_bias = canonical.shadow_bias
        self.highlight_rgb_offset = canonical.highlight_rgb_offset
        self.shadow_rgb_offset = canonical.shadow_rgb_offset
        self.black_level_treatment = canonical.black_level_character
        self.temperature_shift = canonical.global_temperature_intent
        self.tint_shift = canonical.global_tint_intent
        self.black_mist_diffusion_strength = canonical.black_mist_diffusion_strength
        return self

    def get_scene_intent(self, scene_group_id: str) -> Optional[SceneIntent]:
        if isinstance(self.scene_intents, dict):
            return self.scene_intents.get(scene_group_id)
        for s in self.scene_intents:
            if s.scene_group_id == scene_group_id:
                return s
        return None