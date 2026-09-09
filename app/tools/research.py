import os
import json
import time
from typing import List, Optional
from dotenv import load_dotenv
from parallel import Parallel
from google import genai
from google.genai import types

from app.models.analysis import CinematographyResearchResult, SearchCitation, CreativeSpecification

load_dotenv()

def get_parallel_client() -> Optional[Parallel]:
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        return None
    try:
        return Parallel(api_key=api_key)
    except Exception:
        return None

def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)

PRESET_RESEARCH_CONFIGS = {
    "warm_800_negative": {
        "title": "Warm 800 Negative",
        "brief": "35mm tungsten film emulation with golden skin tones, cyan-leaning shadows, gentle highlight roll-off, and moderate saturation.",
        "preferred_source_types": [
            "ASC articles",
            "Kodak technical publications",
            "professional colorist breakdowns"
        ],
        "desired_evidence": [
            "characteristic curve behavior",
            "dye-layer response",
            "color separation principles"
        ],
        "default_principles": [
            "Preserve warm tungsten skin tone rendition while maintaining cyan shadow separation",
            "Gentle highlight roll-off emulation matching 35mm negative shoulder",
            "Controlled color saturation across midtones without spectral clipping"
        ],
        "default_influence": "Guided warm golden highlights, cyan-tinted shadows, and smooth highlight shoulder roll-off."
    },
    "silver_retention": {
        "title": "Silver Retention",
        "brief": "bleach bypass chemical process emulation with increased contrast, desaturated color palette, enhanced silver grain structure, and preserved edge acuity.",
        "preferred_source_types": [
            "film laboratory documentation",
            "cinematography case studies (e.g. Se7en, Saving Private Ryan)"
        ],
        "desired_evidence": [
            "silver halide retention physics",
            "shadow density impact",
            "midtone contrast expansion"
        ],
        "default_principles": [
            "Steepened midtone contrast curve with dense crushed shadows",
            "Reduced global saturation preserving muted tonal separation",
            "Enhanced edge contrast and high-frequency textural acuity"
        ],
        "default_influence": "Guided steep contrast S-curve, desaturated chroma palette, and dense black level."
    },
    "nordic_soft_light": {
        "title": "Nordic Soft Light",
        "brief": "overcast high-latitude naturalism with cool muted palette, soft contrast, neutral skin tones, and gentle shadow roll-off.",
        "preferred_source_types": [
            "Scandinavian cinematography profiles",
            "natural light exterior grading guides"
        ],
        "desired_evidence": [
            "daylight Kelvin response",
            "low-saturation color separation",
            "highlight detail retention"
        ],
        "default_principles": [
            "Soft contrast slope preserving overcast ambient gradient",
            "Cool atmospheric temperature bias with neutral daylight skin tones",
            "Gentle shadow roll-off preventing premature clipping in dark textures"
        ],
        "default_influence": "Guided low-contrast tonal curve, cool atmospheric color temperature, and soft shadow roll-off."
    },
    "neon_nocturne": {
        "title": "Neon Nocturne",
        "brief": "contemporary urban night exterior with saturated practical sources (cyan, magenta, amber), deep controlled shadows, and clean highlight blooming.",
        "preferred_source_types": [
            "neo-noir cinematography analyses",
            "practical neon lighting case studies"
        ],
        "desired_evidence": [
            "spectral peak handling in sRGB/Rec.709",
            "shadow noise floor management",
            "complementary split-toning"
        ],
        "default_principles": [
            "Complementary split-toning separating cyan practicals from warm sodium highlights",
            "Restrained shadow zone chroma preventing digital noise amplification",
            "Controlled highlight roll-off accommodating saturated practical light sources"
        ],
        "default_influence": "Guided cyan/amber split-toning, deep controlled shadow floors, and selective chroma restraint."
    }
}

def get_research_brief(creative_prompt: str) -> dict:
    norm = creative_prompt.lower().replace("-", " ").replace("_", " ")
    if "warm" in norm or "800" in norm or "portra" in norm or "tungsten" in norm:
        return PRESET_RESEARCH_CONFIGS["warm_800_negative"]
    elif "silver" in norm or "bleach" in norm or "bypass" in norm:
        return PRESET_RESEARCH_CONFIGS["silver_retention"]
    elif "nordic" in norm or "soft" in norm or "scandi" in norm or "overcast" in norm:
        return PRESET_RESEARCH_CONFIGS["nordic_soft_light"]
    elif "neon" in norm or "nocturne" in norm or "cyber" in norm:
        return PRESET_RESEARCH_CONFIGS["neon_nocturne"]
    else:
        # Deterministic custom brief
        return {
            "title": creative_prompt.title(),
            "brief": f"Cinematography color grading emulation and lighting design for '{creative_prompt}'.",
            "preferred_source_types": [
                "ASC cinematography articles",
                "professional colorist breakdowns",
                "film stock spectral response documentation"
            ],
            "desired_evidence": [
                "tonal transfer characteristic curves",
                "chromatic separation principles",
                "highlight roll-off and shadow density management"
            ],
            "default_principles": [
                f"Establish visual continuity aligned with '{creative_prompt}'",
                "Maintain natural skin tone reproduction within scene lighting limits",
                "Preserve readable midtones without digital clipping"
            ],
            "default_influence": f"Guided creative color balance and tonal contrast for '{creative_prompt}'."
        }

def research_cinematography_principles(
    creative_prompt: str,
    scene_context: str = "general film scene",
    parallel_client: Optional[Parallel] = None
) -> CinematographyResearchResult:
    if parallel_client is None:
        parallel_client = get_parallel_client()
        
    brief_data = get_research_brief(creative_prompt)
    
    queries = [
        f"{creative_prompt} cinematography color grading lighting",
        f"{creative_prompt} {brief_data['desired_evidence'][0]} {' '.join(brief_data['preferred_source_types'][:2])}"
    ]
    objective = (
        f"Research cinematography techniques, lighting, and color science for: '{creative_prompt}'. "
        f"Brief: {brief_data['brief']} "
        f"Preferred sources: {', '.join(brief_data['preferred_source_types'])}. "
        f"Desired evidence: {', '.join(brief_data['desired_evidence'])}."
    )
    
    if parallel_client is None:
        return CinematographyResearchResult(
            query=queries[0],
            objective=objective,
            sources=[],
            synthesized_principles=brief_data["default_principles"],
            is_grounded=False
        )
        
    try:
        search_res = parallel_client.search(
            search_queries=queries,
            objective=objective,
            mode="fast"
        )
        
        citations: List[SearchCitation] = []
        raw_results = getattr(search_res, "results", []) or []
        
        for r in raw_results[:5]:
            title = getattr(r, "title", "") or "Cinematography Source"
            url = getattr(r, "url", "") or ""
            
            # Primary SDK schema: r.excerpts is list[str]
            raw_excerpts = getattr(r, "excerpts", None)
            excerpt = ""
            if isinstance(raw_excerpts, list) and len(raw_excerpts) > 0:
                excerpt = " ".join([str(e).strip() for e in raw_excerpts if str(e).strip()])
            elif isinstance(raw_excerpts, str) and raw_excerpts.strip():
                excerpt = raw_excerpts.strip()
            else:
                # Compatibility fallback for alternate/older response shapes
                snippet = getattr(r, "snippet", None) or getattr(r, "content", None)
                if snippet and str(snippet).strip():
                    excerpt = str(snippet).strip()
                elif hasattr(r, "highlights") and r.highlights:
                    excerpt = " ".join(r.highlights)
                    
            # Strict grounding criteria: Do NOT substitute title as evidence, must be valid http(s) URL
            if excerpt and url and (url.startswith("http://") or url.startswith("https://")):
                sentences = [s.strip() for s in excerpt.split(". ") if len(s.strip()) > 15]
                if sentences:
                    extracted_princ = sentences[0]
                    if not extracted_princ.endswith("."):
                        extracted_princ += "."
                else:
                    extracted_princ = excerpt[:120].strip() + ("." if not excerpt[:120].strip().endswith(".") else "")

                influence_text = f"Informed by cinematography evidence in '{title}' to guide tonal response and color balance."
                citations.append(SearchCitation(
                    title=str(title).strip() or "Cinematography Reference",
                    url=str(url).strip(),
                    excerpt=str(excerpt)[:400],
                    extracted_principle=extracted_princ,
                    influence=influence_text
                ))
                
        is_grounded = (len(citations) > 0)
        synthesized_principles = [c.extracted_principle for c in citations if c.extracted_principle] if is_grounded else []
        return CinematographyResearchResult(
            query=queries[0],
            objective=objective,
            sources=citations,
            synthesized_principles=synthesized_principles,
            is_grounded=is_grounded
        )
    except Exception as e:
        print(f"[Parallel] Research search failed: {e}")
        # Never fabricate citations on error
        return CinematographyResearchResult(
            query=queries[0],
            objective=objective,
            sources=[],
            synthesized_principles=[],
            is_grounded=False
        )

from app.models.analysis import (
    CinematographyResearchResult,
    SearchCitation,
    CreativeSpecification,
    GlobalLookIntent,
    SceneIntent,
    ShotSemanticAnalysis
)

def build_default_scene_intent(group_id: str, semantic: Optional[ShotSemanticAnalysis] = None) -> SceneIntent:
    if semantic:
        tod = (semantic.time_of_day or "").lower()
        env = (semantic.lighting_environment or "").lower()
        exp = (semantic.exposure_assessment or "balanced").lower()
        
        if "night" in tod or "night" in env or "dark" in env or "low_key" in exp:
            lighting_class = "low_key_night"
            exposure_class = "low_key"
            shadow_sat_ceiling = 0.80
            midtone_sat_ceiling = 1.10
            exp_bounds = [-0.6, 0.6]
            rationale = "Preserve low-key night exposure and practical highlights; restrain chroma in deep shadows."
            anchors = ["practical_lights", "deep_shadows", "faces"]
        elif "golden" in tod or "sunset" in tod or "golden" in env:
            lighting_class = "golden_hour"
            exposure_class = "balanced"
            shadow_sat_ceiling = 0.95
            midtone_sat_ceiling = 1.25
            exp_bounds = [-0.8, 0.8]
            rationale = "Preserve warm golden-hour ambience with natural contrast."
            anchors = ["warm_highlights", "natural_skin"]
        else:
            lighting_class = "daylight"
            exposure_class = exp if exp in ["balanced", "underexposed", "overexposed", "high_key"] else "balanced"
            shadow_sat_ceiling = 0.95
            midtone_sat_ceiling = 1.30
            exp_bounds = [-1.0, 1.0]
            rationale = "Maintain balanced daylight exposure and faithful tonal distribution."
            anchors = ["midtone_contrast", "neutral_whites"]
            
        return SceneIntent(
            scene_group_id=group_id,
            lighting_class=lighting_class,
            exposure_class=exposure_class,
            scene_mood="cinematic",
            source_relative_exposure_bounds=exp_bounds,
            target_tonal_rules=["Preserve readable midtones", "Prevent unnatural shadow color tint"],
            shadow_saturation_ceiling=shadow_sat_ceiling,
            midtone_saturation_ceiling=midtone_sat_ceiling,
            contrast_trim_bounds=[0.85, 1.15],
            saturation_trim_bounds=[0.70, 1.20],
            protected_visual_anchors=anchors,
            confidence=0.90,
            concise_rationale=rationale
        )
    return SceneIntent(
        scene_group_id=group_id,
        lighting_class="daylight",
        exposure_class="balanced",
        scene_mood="natural",
        source_relative_exposure_bounds=[-1.0, 1.0],
        target_tonal_rules=["Preserve dynamic range"],
        shadow_saturation_ceiling=0.95,
        midtone_saturation_ceiling=1.30,
        contrast_trim_bounds=[0.85, 1.15],
        saturation_trim_bounds=[0.70, 1.20],
        protected_visual_anchors=["neutral_whites"],
        confidence=0.85,
        concise_rationale="Maintain balanced exposure and natural contrast."
    )

def synthesize_creative_specification(
    creative_prompt: str,
    research_result: CinematographyResearchResult,
    scene_analyses: Optional[List[ShotSemanticAnalysis]] = None,
    genai_client: Optional[genai.Client] = None,
    max_retries: int = 3
) -> CreativeSpecification:
    if genai_client is None:
        genai_client = get_genai_client()
        
    if research_result.is_grounded and research_result.sources:
        sources_text = "\n".join([f"- [{s.title}]({s.url}): {s.excerpt}" for s in research_result.sources])
        research_section = f"Cinematography research from Parallel:\n{sources_text}"
    else:
        research_section = "Parallel research: Grounding unavailable. Rely on expert digital intermediate color science principles."

    # Identify unique scene groups
    scene_groups_info = ""
    group_map = {}
    if scene_analyses:
        for s in scene_analyses:
            if s.scene_group_id not in group_map:
                group_map[s.scene_group_id] = s
        lines = []
        for gid, s in group_map.items():
            lines.append(f"- Scene Group '{gid}': Setting='{s.scene_description}', Lighting='{s.lighting_environment}', Time='{s.time_of_day}', Exposure='{s.exposure_assessment}'")
        scene_groups_info = "Detected Sequence Scene Groups:\n" + "\n".join(lines)
    else:
        scene_groups_info = "Detected Sequence Scene Groups: Single scene group 'group_1'."

    brief_data = get_research_brief(creative_prompt)
    brief_section = (
        f"Aesthetic Directive: {brief_data['title']}\n"
        f"Cinematography Brief: {brief_data['brief']}\n"
        f"Key Evidence Focus: {', '.join(brief_data['desired_evidence'])}\n"
    )

    prompt = f"""You are a master digital intermediate (DI) supervisor and scene planning colorist.
A filmmaker has requested the following creative color direction:
User Prompt: "{creative_prompt}"

{brief_section}

{research_section}

{scene_groups_info}

Synthesize this into a structured CreativeSpecification with both GlobalLookIntent and per-scene SceneIntent:
1. Global Look Intent (shared sequence-wide creative identity):
   - contrast_intent: 0.90 to 1.35.
   - saturation_intent: 0.70 to 1.40.
   - temperature_shift: -25.0 to +25.0.
   - tint_shift: -15.0 to +15.0.
   - highlight_bias & shadow_bias (e.g. warm golden, cool slate, neon cyan).
   - highlight_rgb_offset & shadow_rgb_offset in [-0.15, 0.15] [B, G, R].
2. Scene Intents: For every scene group listed above, provide a SceneIntent item in the scene_intents list with matching scene_group_id:
   - lighting_class: daylight, golden_hour, low_key_night, practical_night, interior_tungsten, etc.
   - exposure_class: balanced, low_key, high_key, underexposed, etc.
   - shadow_saturation_ceiling: 0.75-0.85 for dark/night scenes, 0.95-1.10 for daylight.
   - midtone_saturation_ceiling: 1.10-1.35.
   - concise_rationale: 1-sentence colorist lighting intent for this scene.
3. Extract 2-4 key cinematography principles.
4. Output strictly conforming JSON matching CreativeSpecification.
"""
    
    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash",
        "gemini-3.6-flash"
    ]
    
    last_error = None
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CreativeSpecification,
                        temperature=0.2
                    )
                )
                spec: CreativeSpecification = response.parsed
                spec.synthesis_mode = "grounded" if research_result.is_grounded else "ungrounded"
                spec.citations = research_result.sources if research_result.is_grounded else []

                # Ensure global_look is populated
                if not spec.global_look:
                    spec.global_look = GlobalLookIntent(
                        look_title=spec.look_title,
                        base_contrast=spec.contrast_intent,
                        base_saturation=spec.saturation_intent,
                        shadow_bias=spec.shadow_bias,
                        highlight_bias=spec.highlight_bias,
                        highlight_rgb_offset=spec.highlight_rgb_offset,
                        shadow_rgb_offset=spec.shadow_rgb_offset,
                        black_level_character=spec.black_level_treatment,
                        global_temperature_intent=spec.temperature_shift,
                        global_tint_intent=spec.tint_shift,
                        black_mist_diffusion_strength=spec.black_mist_diffusion_strength
                    )

                # Ensure every detected scene group has a SceneIntent
                existing_gids = {s.scene_group_id for s in spec.scene_intents}
                if group_map:
                    for gid, sem in group_map.items():
                        if gid not in existing_gids:
                            spec.scene_intents.append(build_default_scene_intent(gid, sem))
                            existing_gids.add(gid)
                elif not spec.scene_intents:
                    spec.scene_intents.append(build_default_scene_intent("group_1"))

                spec.normalize_canonical_look()
                return spec
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "prepayment credits are depleted" in err_str.lower():
                    break
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break

    # If all models in cascade fail, report and use deterministic neutral fallback
    fallback_reason = str(last_error) if last_error else "Model unavailable"
    if "prepayment credits are depleted" in fallback_reason.lower():
        fallback_reason = (
            "Google AI Studio prepayment credits depleted ($0 balance). "
            "Generate a free API key at https://aistudio.google.com/app/apikey in a new project."
        )

    print(f"[Synthesizer Warning] All creative synthesis models failed: {fallback_reason}. Using neutral baseline.")
    
    fallback_global = GlobalLookIntent(
        look_title="Neutral Photographic Baseline",
        base_contrast=1.0,
        base_saturation=1.0,
        shadow_bias="neutral",
        highlight_bias="neutral",
        highlight_rgb_offset=[0.0, 0.0, 0.0],
        shadow_rgb_offset=[0.0, 0.0, 0.0],
        black_level_character="neutral"
    )

    fallback_scene_intents = []
    if group_map:
        for gid, sem in group_map.items():
            fallback_scene_intents.append(build_default_scene_intent(gid, sem))
    else:
        fallback_scene_intents.append(build_default_scene_intent("group_1"))

    return CreativeSpecification(
        look_title="Neutral Photographic Baseline",
        target_aesthetic=creative_prompt,
        synthesis_mode="fallback",
        fallback_reason=fallback_reason,
        contrast_intent=1.0,
        saturation_intent=1.0,
        highlight_bias="neutral",
        shadow_bias="neutral",
        highlight_rgb_offset=[0.0, 0.0, 0.0],
        shadow_rgb_offset=[0.0, 0.0, 0.0],
        black_level_treatment="neutral",
        temperature_shift=0.0,
        tint_shift=0.0,
        black_mist_diffusion_strength=0.0,
        cinematography_principles=["Preserve source dynamic range", "Neutral color reproduction"],
        citations=research_result.sources if research_result.is_grounded else [],
        global_look=fallback_global,
        scene_intents=fallback_scene_intents
    )