from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class InputProfileAssessment(BaseModel):
    shot_id: str
    selected_profile: str
    metadata_hint: str = "unknown"
    signal_class_hint: Literal["display_ready", "log_like", "ambiguous"] = "display_ready"
    confidence: float = 0.5
    reasons: List[str] = Field(default_factory=list)
    profile_mismatch_warning: bool = False
    warning_message: Optional[str] = None

class NormalizationValidationResult(BaseModel):
    shot_id: str
    state: Literal["NORMALIZATION_VERIFIED", "PROFILE_CONFIRMATION_REQUIRED", "NORMALIZATION_WARNING_OVERRIDDEN", "NORMALIZATION_FAILED"]
    passed: bool
    reason: str
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)

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

class CinematographyResearchResult(BaseModel):
    query: str
    objective: str
    sources: List[SearchCitation]
    synthesized_principles: List[str] = Field(default_factory=list, description="Core cinematography principles extracted from sources")
    is_grounded: bool = Field(True, description="True if Parallel returned genuine verified research sources")

class CreativeSpecification(BaseModel):
    look_title: str
    target_aesthetic: str
    synthesis_mode: Literal["grounded", "ungrounded", "fallback"] = Field("ungrounded", description="Provenance of the creative specification")
    contrast_intent: float = Field(1.0, ge=0.5, le=1.8, description="Target contrast multiplier")
    saturation_intent: float = Field(1.0, ge=0.0, le=2.0, description="Target saturation multiplier")
    highlight_bias: str = Field("neutral", description="e.g. warm amber, neutral, soft golden")
    shadow_bias: str = Field("neutral", description="e.g. cool teal, deep blue, neutral")
    black_level_treatment: str = Field("neutral", description="e.g. filmic lifted, deep crushed, neutral")
    temperature_shift: float = Field(0.0, ge=-50.0, le=50.0, description="Creative temperature bias")
    tint_shift: float = Field(0.0, ge=-50.0, le=50.0, description="Creative tint bias")
    black_mist_diffusion_strength: float = Field(0.0, ge=0.0, le=1.0, description="Black Mist diffusion emulation intensity")
    cinematography_principles: List[str] = Field(default_factory=list)
    citations: List[SearchCitation] = Field(default_factory=list)