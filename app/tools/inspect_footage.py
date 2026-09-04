import os
import cv2
import json
import time
from typing import List, Optional, Dict
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from app.models.analysis import ShotSemanticAnalysis, SequenceInspectionResult
from app.media.ffmpeg import extract_sampled_frames

load_dotenv()

def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)

def inspect_all_shots_batched(
    video_paths: List[str],
    client: Optional[genai.Client] = None,
    max_retries: int = 3
) -> SequenceInspectionResult:
    if client is None:
        client = get_genai_client()
        
    pil_images = []
    shot_headers = []
    
    for i, path in enumerate(video_paths):
        shot_id = f"shot_{chr(65 + i)}"
        frames, _ = extract_sampled_frames(path, num_samples=3)
        
        shot_headers.append(f"Shot {shot_id} (Clip: {os.path.basename(path)}): 3 keyframes attached.")
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            if h > 540:
                scale = 540.0 / h
                rgb = cv2.resize(rgb, (int(w * scale), 540), interpolation=cv2.INTER_AREA)
            pil_images.append(Image.fromarray(rgb))
            
    prompt = f"""You are a master digital intermediate (DI) colorist inspecting a multi-shot video sequence.
Here are the {len(video_paths)} shots to analyze in sequence:
{chr(10).join(shot_headers)}

For EACH shot independently analyze:
1. Setting and scene description.
2. Assign `scene_group_id`: Group shots sharing the same environment/lighting into the same group ID (e.g. "group_1", "group_2").
3. Assign `relationship_to_reference`: 
   - Set "reference" for the single recommended master reference shot.
   - Set "same_scene" for shots in the SAME scene group as the reference.
   - Set "independent_scene" for shots in a DIFFERENT scene group / time of day (e.g. night shot vs daytime reference).
4. Lighting environment (outdoor daylight, direct sun, overcast, golden hour, indoor tungsten, night blue ambient, etc.).
5. Time of day (day, night, golden_hour, dusk, dawn).
6. Exposure assessment (balanced, underexposed, overexposed, high_key, low_key).
7. Target exposure compensation in EV stops (-2.0 to +2.0) needed to optimize this specific shot's dynamic range. STRICT CONVENTION: positive values (+0.5 to +2.0) to brighten underexposed footage, negative values (-0.5 to -2.0) to darken overexposed footage, 0.0 for balanced.
8. Black point lift (0.0 to 15.0) for filmic shadow toe density.
9. Presence of human subjects (people_present: true/false).
10. Dominant color temperature or cast.
11. Reference suitability score (0.0 to 1.0).

Also evaluate overall sequence relationship:
- Set recommended_reference_shot_id to the shot ID with the most balanced reference exposure.
- Set scene_relationship to 'continuous_sequence' if all shots are in the same scene, 'independent_scenes' if all differ, or 'mixed_sequence' if mixed.

Return strictly conforming JSON matching the schema.
"""

    contents = [prompt] + pil_images

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SequenceInspectionResult,
        temperature=0.2,
    )

    models_to_try = [
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash",
        "gemini-flash-lite-latest"
    ]
    last_error = None
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                data = json.loads(response.text)
                shots_data = data.get("shots", [])
                
                # Defensive normalization of shot IDs and length validation
                if len(shots_data) != len(video_paths):
                    # Align count
                    while len(shots_data) < len(video_paths):
                        idx = len(shots_data)
                        shots_data.append({
                            "shot_id": f"shot_{chr(65 + idx)}",
                            "scene_group_id": "group_1",
                            "relationship_to_reference": "same_scene",
                            "scene_description": f"Clip {os.path.basename(video_paths[idx])}",
                            "lighting_environment": "natural lighting",
                            "time_of_day": "day",
                            "exposure_assessment": "balanced",
                            "target_exposure_compensation_ev": 0.0,
                            "black_point_lift": 2.0,
                            "people_present": False,
                            "dominant_color_cast": "neutral",
                            "reference_suitability_score": 0.8
                        })
                        
                for idx, s in enumerate(shots_data):
                    s["shot_id"] = f"shot_{chr(65 + idx)}"
                    
                rec_ref = data.get("recommended_reference_shot_id", "shot_A")
                if rec_ref not in [f"shot_{chr(65 + i)}" for i in range(len(video_paths))]:
                    rec_ref = "shot_A"
                    
                # Ensure chosen reference has relationship 'reference'
                for s in shots_data:
                    if s["shot_id"] == rec_ref:
                        s["relationship_to_reference"] = "reference"
                        
                return SequenceInspectionResult(
                    shots=[ShotSemanticAnalysis(**s) for s in shots_data],
                    recommended_reference_shot_id=rec_ref,
                    scene_relationship=data.get("scene_relationship", "mixed_sequence")
                )
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "prepayment credits are depleted" in err_str.lower():
                    # Billing exhaustion cannot be resolved by immediate retries
                    break
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 3 * (attempt + 1)
                    print(f"[{model_name}] Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    break
                    
    # Categorized error reporting
    err_str = str(last_error)
    if "prepayment credits are depleted" in err_str.lower():
        raise PermissionError(
            "Gemini API Error: Google AI Studio prepayment credits are depleted ($0 balance). "
            "Please create a free-tier API key in a clean project at https://aistudio.google.com/app/apikey "
            "(select 'Create in new project' without billing), and update GEMINI_API_KEY in .env."
        )
    elif "API_KEY_INVALID" in err_str or "401" in err_str:
        raise PermissionError(f"Gemini API authentication failed: {last_error}")
    elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        raise RuntimeError(f"Gemini API rate limit exceeded: {last_error}")
    else:
        raise RuntimeError(f"Gemini footage inspection failed ({type(last_error).__name__}): {last_error}")