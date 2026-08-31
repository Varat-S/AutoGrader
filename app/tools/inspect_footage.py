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

from app.models.analysis import ShotSemanticAnalysis
from app.media.ffmpeg import extract_sampled_frames

load_dotenv()

class SequenceInspectionResult(BaseModel):
    shots: List[ShotSemanticAnalysis]
    recommended_reference_shot_id: str = Field(..., description="ID of the optimal technical reference shot")
    scene_relationship: str = Field("independent_scenes", description="continuous_sequence or independent_scenes")

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
        frames, _ = extract_sampled_frames(path, fractions=[0.25, 0.50, 0.75])
        
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
2. Lighting environment (outdoor daylight, direct sun, overcast, golden hour, indoor tungsten, twilight, etc.).
3. Time of day (day, night, golden_hour, dusk, dawn).
4. Exposure assessment (balanced, underexposed, overexposed, high_key, low_key).
5. Target exposure compensation in EV stops (-2.0 to +2.0) needed to optimize this specific shot's dynamic range without blowing highlights.
6. Black point lift (0.0 to 15.0) to emulate soft filmic shadow density / Black Mist diffusion.
7. Presence of human subjects, faces, or skin tones (if present, skin_protection_required=True).
8. Dominant color temperature or cast.
9. Reference suitability score (0.0 to 1.0).

Also evaluate the sequence relationship:
- Set scene_relationship='continuous_sequence' if shots are camera angles of the SAME scene/time of day.
- Set scene_relationship='independent_scenes' if shots are taken in different environments/times of day.
- Set recommended_reference_shot_id to the shot ID with the most balanced exposure/lighting.

Return strictly conforming JSON matching the schema.
"""

    contents = [prompt] + pil_images

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SequenceInspectionResult,
        temperature=0.2,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
                config=config,
            )
            data = json.loads(response.text)
            # Ensure shot_ids are assigned
            for idx, s in enumerate(data.get("shots", [])):
                s["shot_id"] = f"shot_{chr(65 + idx)}"
            return SequenceInspectionResult(**data)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 15 * (attempt + 1)
                print(f"[Gemini] Rate limit hit. Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
                
    raise RuntimeError("Gemini API rate limit exceeded after maximum retries. Please wait 30 seconds.")