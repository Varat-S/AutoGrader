import os
import cv2
import json
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import io

from app.models.analysis import ShotSemanticAnalysis
from app.media.ffmpeg import extract_sampled_frames

load_dotenv()

def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)

def inspect_footage_semantics(
    video_path: str,
    shot_id: str = "shot",
    frames: Optional[List[np.ndarray]] = None,
    client: Optional[genai.Client] = None
) -> ShotSemanticAnalysis:
    if client is None:
        client = get_genai_client()
        
    if frames is None:
        frames, _ = extract_sampled_frames(video_path, fractions=[0.25, 0.50, 0.75])
        
    pil_images = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        if h > 720:
            scale = 720.0 / h
            rgb = cv2.resize(rgb, (int(w * scale), 720), interpolation=cv2.INTER_AREA)
        pil_images.append(Image.fromarray(rgb))
        
    prompt = f"""You are a master digital intermediate (DI) colorist and Director of Photography inspecting video footage for shot '{shot_id}'.

Analyze this specific scene independently:
1. Setting and visual content.
2. Lighting environment (e.g. outdoor natural daylight, direct sun, overcast, golden hour, indoor tungsten, twilight).
3. Time of day (day, night, golden_hour, dusk, dawn).
4. Exposure assessment (balanced, underexposed, overexposed, high_key, low_key).
5. Target exposure compensation in EV stops (-2.0 to +2.0) needed to optimize this specific shot's dynamic range without blowing highlights or losing mood.
6. Black point lift (0.0 to 15.0) to emulate soft filmic shadow density / Black Mist diffusion.
7. Presence of human subjects, faces, or skin tones (if present, skin_protection_required=True).
8. Dominant color temperature or cast.
9. Practical / intentional light sources.
10. Reference suitability score (0.0 to 1.0).

Return strictly conforming JSON matching the schema.
"""

    contents = [prompt] + pil_images

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ShotSemanticAnalysis,
        temperature=0.2,
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=config,
    )

    data = json.loads(response.text)
    data["shot_id"] = shot_id
    return ShotSemanticAnalysis(**data)