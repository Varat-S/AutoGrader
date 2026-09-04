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

def research_cinematography_principles(
    creative_prompt: str,
    scene_context: str = "general film scene",
    parallel_client: Optional[Parallel] = None
) -> CinematographyResearchResult:
    if parallel_client is None:
        parallel_client = get_parallel_client()
        
    queries = [
        f"{creative_prompt} cinematography color grading lighting",
        f"{creative_prompt} film stock palette colorist breakdown"
    ]
    objective = f"Research cinematography techniques, film stock color response, and colorist principles for: '{creative_prompt}' in a {scene_context}."
    
    if parallel_client is None:
        return CinematographyResearchResult(
            query=queries[0],
            objective=objective,
            sources=[],
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
                    
            # Strict grounding criteria: Do NOT substitute title as evidence
            if excerpt and url and (url.startswith("http://") or url.startswith("https://")):
                citations.append(SearchCitation(
                    title=str(title).strip() or "Cinematography Reference",
                    url=str(url).strip(),
                    excerpt=str(excerpt)[:400]
                ))
                
        is_grounded = (len(citations) > 0)
        return CinematographyResearchResult(
            query=queries[0],
            objective=objective,
            sources=citations,
            is_grounded=is_grounded
        )
    except Exception as e:
        print(f"[Parallel] Research search failed: {e}")
        # Never fabricate citations on error
        return CinematographyResearchResult(
            query=queries[0],
            objective=objective,
            sources=[],
            is_grounded=False
        )

def synthesize_creative_specification(
    creative_prompt: str,
    research_result: CinematographyResearchResult,
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
        
    prompt = f"""You are a master digital intermediate (DI) colorist.
A filmmaker has requested the following creative color direction:
User Prompt: "{creative_prompt}"

{research_section}

Synthesize this into a technical CreativeSpecification:
1. Translate artistic descriptions into numeric values:
   - contrast_intent: 0.90 to 1.30 (if Black-Mist-inspired tonal response is requested, soften contrast slightly to ~0.95 - 1.05).
   - saturation_intent: 0.70 to 1.30.
   - temperature_shift: -25.0 to +25.0.
   - tint_shift: -15.0 to +15.0.
   - black_mist_diffusion_strength: 0.0 to 1.0 (tonal toe lift parameter).
2. Explicitly specify highlight bias (e.g. warm amber, cool cyan), shadow bias (e.g. cool slate, deep teal), and black level treatment (e.g. filmic lifted, deep crushed).
3. Extract 2-4 key cinematography principles.
4. Output strictly conforming JSON matching the schema.
"""
    
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
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
            return spec
        except Exception as e:
            if attempt == max_retries - 1:
                # Deterministic neutral fallback specification
                return CreativeSpecification(
                    look_title="Neutral Photographic Baseline",
                    target_aesthetic=creative_prompt,
                    synthesis_mode="fallback",
                    contrast_intent=1.0,
                    saturation_intent=1.0,
                    highlight_bias="neutral",
                    shadow_bias="neutral",
                    black_level_treatment="neutral",
                    temperature_shift=0.0,
                    tint_shift=0.0,
                    black_mist_diffusion_strength=0.0,
                    cinematography_principles=["Preserve source dynamic range", "Neutral color reproduction"],
                    citations=research_result.sources if research_result.is_grounded else []
                )
            time.sleep(1.0)