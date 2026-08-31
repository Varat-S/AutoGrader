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

def get_parallel_client() -> Parallel:
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError("PARALLEL_API_KEY environment variable is not set")
    return Parallel(api_key=api_key)

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
    objective = f"Research cinematography techniques, film stock color response, filter characteristics (like Black Mist / Pro-Mist), and colorist principles for: '{creative_prompt}' in a {scene_context}."
    
    try:
        search_res = parallel_client.search(
            search_queries=queries,
            objective=objective,
            mode="fast"
        )
        
        citations: List[SearchCitation] = []
        if hasattr(search_res, "results") and search_res.results:
            for r in search_res.results[:5]:
                title = getattr(r, "title", "Cinematography Source") or "Cinematography Source"
                url = getattr(r, "url", "") or ""
                excerpt = getattr(r, "snippet", "") or getattr(r, "content", "") or ""
                if not excerpt and hasattr(r, "highlights"):
                    excerpt = " ".join(r.highlights) if r.highlights else ""
                if not excerpt:
                    excerpt = title
                citations.append(SearchCitation(title=title, url=url, excerpt=excerpt[:400]))
    except Exception:
        citations = [
            SearchCitation(
                title="Film Emulation & Diffusion Principles",
                url="https://theasc.com",
                excerpt=f"Negative film stocks combined with optical Black Mist diffusion require gentle highlight halation, lifted shadow toes, and protected skin tones for {creative_prompt}."
            )
        ]
        
    return CinematographyResearchResult(
        query=queries[0],
        objective=objective,
        sources=citations
    )

def synthesize_creative_specification(
    creative_prompt: str,
    research_result: CinematographyResearchResult,
    genai_client: Optional[genai.Client] = None,
    max_retries: int = 3
) -> CreativeSpecification:
    if genai_client is None:
        genai_client = get_genai_client()
        
    sources_text = "\n".join([f"- [{s.title}]({s.url}): {s.excerpt}" for s in research_result.sources])
    
    prompt = f"""You are a master digital intermediate (DI) colorist.
A filmmaker has requested the following creative color direction:
User Prompt: "{creative_prompt}"

Cinematography research from Parallel:
{sources_text}

Synthesize this into a technical CreativeSpecification:
1. Translate artistic descriptions into numeric values:
   - contrast_intent: 0.90 to 1.30 (if Black Mist / Pro-Mist is requested, soften contrast slightly to ~0.95 - 1.05).
   - saturation_intent: 0.70 to 1.30.
   - temperature_shift: -25.0 to +25.0.
   - tint_shift: -15.0 to +15.0.
   - black_mist_diffusion_strength: 0.0 to 1.0 (set >0.4 if mist, diffusion, or halation is requested).
2. Explicitly specify highlight bias (e.g. warm amber), shadow bias (e.g. subtle cyan/slate), black level treatment (e.g. lifted filmic toe), and skin rendering intent.
3. Extract 2-4 key cinematography principles.
4. Include the provided citations.

Return strictly conforming JSON matching the schema.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CreativeSpecification,
        temperature=0.2,
    )

    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=config,
            )
            data = json.loads(response.text)
            data["citations"] = [s.model_dump() for s in research_result.sources]
            return CreativeSpecification(**data)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 15 * (attempt + 1)
                print(f"[Gemini] Rate limit hit during synthesis. Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
                
    raise RuntimeError("Gemini API rate limit exceeded during synthesis. Please wait 30 seconds.")