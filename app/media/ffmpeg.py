import subprocess
import json
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

_CACHED_FFMPEG = None
_CACHED_FFPROBE = None

def get_ffmpeg_binary() -> str:
    global _CACHED_FFMPEG
    if _CACHED_FFMPEG and os.path.exists(_CACHED_FFMPEG):
        return _CACHED_FFMPEG
        
    # 1. System PATH
    found = shutil.which('ffmpeg')
    if found:
        _CACHED_FFMPEG = found
        return found
        
    # 2. WinGet package directory on Windows
    winget_dir = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages'
    if winget_dir.exists():
        candidates = list(winget_dir.glob('**/ffmpeg.exe'))
        if candidates:
            _CACHED_FFMPEG = str(candidates[0])
            return _CACHED_FFMPEG
            
    return 'ffmpeg'

def get_ffprobe_binary() -> str:
    global _CACHED_FFPROBE
    if _CACHED_FFPROBE and os.path.exists(_CACHED_FFPROBE):
        return _CACHED_FFPROBE
        
    # 1. System PATH
    found = shutil.which('ffprobe')
    if found:
        _CACHED_FFPROBE = found
        return found
        
    # 2. WinGet package directory on Windows
    winget_dir = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages'
    if winget_dir.exists():
        candidates = list(winget_dir.glob('**/ffprobe.exe'))
        if candidates:
            _CACHED_FFPROBE = str(candidates[0])
            return _CACHED_FFPROBE
            
    return 'ffprobe'

def run_subprocess(cmd: List[str], timeout_sec: int = 120) -> subprocess.CompletedProcess:
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False
        )
        if res.returncode != 0:
            cmd_str = ' '.join(cmd)
            raise RuntimeError(f'Command failed (code {res.returncode}): {cmd_str}\nStderr: {res.stderr}')
        return res
    except subprocess.TimeoutExpired as e:
        cmd_str = ' '.join(cmd)
        raise TimeoutError(f'Command timed out after {timeout_sec}s: {cmd_str}') from e

def probe_video(video_path: str) -> Dict[str, Any]:
    cmd = [
        get_ffprobe_binary(),
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration,codec_name,pix_fmt',
        '-show_entries', 'format=duration,size',
        '-of', 'json',
        video_path
    ]
    res = run_subprocess(cmd)
    data = json.loads(res.stdout)
    
    stream = data.get('streams', [{}])[0] if data.get('streams') else {}
    fmt = data.get('format', {})
    
    width = int(stream.get('width', 1920))
    height = int(stream.get('height', 1080))
    
    r_fps = stream.get('r_frame_rate', '30/1')
    try:
        num, den = map(int, r_fps.split('/'))
        fps = num / den if den > 0 else 30.0
    except Exception:
        fps = 30.0
        
    duration = float(stream.get('duration') or fmt.get('duration') or 0.0)
    
    return {
        'path': video_path,
        'width': width,
        'height': height,
        'fps': round(fps, 2),
        'duration_sec': round(duration, 2),
        'codec': stream.get('codec_name', 'unknown'),
        'pix_fmt': stream.get('pix_fmt', 'unknown')
    }

def extract_sampled_frames(video_path: str, fractions: List[float] = [0.25, 0.50, 0.75]) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Failed to open video file: {video_path}')
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if total_frames > 0 else 1.0
    
    frames = []
    timestamps = []
    
    for frac in fractions:
        target_frame = int(round(total_frames * frac))
        target_frame = max(0, min(target_frame, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            timestamps.append(round(target_frame / fps, 2))
            
    cap.release()
    
    if not frames:
        raise ValueError(f'Could not extract any frames from {video_path}')
        
    return frames, timestamps

def generate_proxy(input_path: str, output_path: str, max_height: int = 720) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        get_ffmpeg_binary(),
        '-y',
        '-i', input_path,
        '-vf', f'scale=-2:min({max_height},ih)',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    run_subprocess(cmd)
    return output_path

def apply_lut_and_render(
    input_path: str,
    lut_path: str,
    output_path: str,
    preset: str = 'fast',
    crf: int = 18
) -> str:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    formatted_lut_path = str(Path(lut_path).resolve().as_posix()).replace(':', '\\:')
    
    cmd = [
        get_ffmpeg_binary(),
        '-y',
        '-i', input_path,
        '-vf', f'lut3d=file=\'{formatted_lut_path}\'',
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        output_path
    ]
    try:
        run_subprocess(cmd)
    except Exception:
        cmd_fallback = [
            get_ffmpeg_binary(),
            '-y',
            '-i', input_path,
            '-vf', f'lut3d=file=\'{formatted_lut_path}\'',
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-an',
            output_path
        ]
        run_subprocess(cmd_fallback)
        
    return output_path
