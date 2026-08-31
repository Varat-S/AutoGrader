import os
from typing import Tuple
from app.models.grade import ColorGradeParams
from app.media.lut import generate_3d_cube_lut
from app.media.ffmpeg import apply_lut_and_render

def render_grade(
    source_video_path: str,
    params: ColorGradeParams,
    output_video_path: str,
    output_lut_path: str,
    lut_size: int = 33,
    is_preview: bool = False,
    is_log: bool = False
) -> Tuple[str, str]:
    # 1. Bake 3D LUT (with Log CST if applicable)
    generate_3d_cube_lut(params, output_lut_path, size=lut_size, is_log=is_log)
    
    # 2. Render with FFmpeg
    preset = "veryfast" if is_preview else "medium"
    crf = 22 if is_preview else 18
    apply_lut_and_render(source_video_path, output_lut_path, output_video_path, preset=preset, crf=crf)
    
    return output_video_path, output_lut_path