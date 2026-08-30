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
        
    # Convert OpenCV BGR frames to PIL RGB Images for Gemini
    pil_images = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        # Resize to max 720p height for token efficiency
        h, w, _ = rgb.shape
        if h > 720:
            scale = 720.0 / h
            rgb = cv2.resize(rgb, (int(w * scale), 720), interpolation=cv2.INTER_AREA)
        pil_images.append(Image.fromarray(rgb))
        
    prompt = f"""You are an expert film colorist and cinematographer analyzing video footage for shot-to-shot color grading.
Inspect the provided sequential keyframes from video shot '{shot_id}'.

Analyze:
1. Scene setting and visual content.
2. Lighting environment (e.g. outdoor natural daylight, overcast, direct sunlight, golden hour, indoor tungsten).
3. Time of day.
4. Presence of human subjects, faces, or skin tones. If present, set skin_protection_required=True.
5. Dominant color temperature or cast (warm, cool, magenta, green, neutral).
6. Practical / intentional light sources that should not be aggressively neutralized.
7. Likely neutral reference objects (e.g. pavement, gray walls, white shirts).
8. Suitability score (0.0 to 1.0) to serve as the master technical reference shot.

Return your analysis strictly matching the requested JSON schema.
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

    # Parse validated JSON into Pydantic model
    data = json.loads(response.text)
    data["shot_id"] = shot_id
    return ShotSemanticAnalysis(**data)