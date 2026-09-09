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
        '-show_entries', 'stream=width,height,r_frame_rate,duration,codec_name,pix_fmt,color_primaries,color_transfer,color_space,color_range,bits_per_raw_sample',
        '-show_entries', 'format=duration,size',
        '-of', 'json',
        video_path
    ]
    res = run_subprocess(cmd)
    data = json.loads(res.stdout)
    
    streams = data.get('streams', [])
    if not streams:
        raise ValueError(f"No valid video stream found in: {video_path}")
    stream = streams[0]
    fmt = data.get('format', {})
    
    width = int(stream.get('width', 0))
    height = int(stream.get('height', 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions {width}x{height} in {video_path}")
        
    if width * height > 4096 * 2160:
        raise ValueError(f"Video resolution {width}x{height} exceeds maximum supported 4K UHD ceiling (4096x2160).")
    
    r_fps = stream.get('r_frame_rate', '30/1')
    try:
        num, den = map(int, r_fps.split('/'))
        fps = num / den if den > 0 else 30.0
    except Exception:
        fps = 30.0
        
    duration = float(stream.get('duration') or fmt.get('duration') or 0.0)
    if duration <= 0.0:
        raise ValueError(f"Invalid duration {duration}s in {video_path}")
    if duration > 600.0:
        raise ValueError(f"Video duration {duration:.1f}s exceeds maximum allowed 600s ceiling.")
        
    # Verify at least one frame decodes successfully
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video container: {video_path}")
    ret, test_frame = cap.read()
    cap.release()
    if not ret or test_frame is None or test_frame.size == 0:
        raise ValueError(f"Corrupted video stream: failed to decode any video frames in {video_path}")
    
    return {
        'path': video_path,
        'width': width,
        'height': height,
        'fps': round(fps, 2),
        'duration_sec': round(duration, 2),
        'codec': stream.get('codec_name', 'unknown'),
        'pix_fmt': stream.get('pix_fmt', 'unknown'),
        'bits_per_raw_sample': stream.get('bits_per_raw_sample', 'unknown'),
        'color_primaries': stream.get('color_primaries', 'unknown'),
        'color_transfer': stream.get('color_transfer', 'unknown'),
        'color_space': stream.get('color_space', 'unknown'),
        'color_range': stream.get('color_range', 'unknown')
    }

def extract_sampled_frames(
    video_path: str,
    fractions: Optional[List[float]] = None,
    num_samples: int = 6
) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Failed to open video file: {video_path}')
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if total_frames > 0 else 1.0
    
    if fractions is None:
        if num_samples <= 1:
            sample_fracs = [0.5]
        else:
            sample_fracs = [float(i) / float(num_samples + 1) for i in range(1, num_samples + 1)]
    else:
        sample_fracs = fractions
        
    frames = []
    timestamps = []
    
    for frac in sample_fracs:
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
    
    # Try stream copy for audio first
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
        # Fallback 1: Re-encode audio to AAC
        cmd[-2] = 'aac'
        cmd.insert(-1, '-b:a')
        cmd.insert(-1, '192k')
        try:
            run_subprocess(cmd)
        except Exception:
            # Fallback 2: Render without audio (-an)
            cmd_silent = [
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
            run_subprocess(cmd_silent)
            
    return output_path

def generate_matched_browser_proxies(
    source_path: str,
    lut_path: str,
    before_proxy_path: str,
    after_proxy_path: str,
    max_height: int = 720
) -> Tuple[str, str]:
    """Generates strictly matched before/after browser proxy videos for synchronized side-by-side display.
    Guarantees identical resolution, frame rate, libx264 profile, pixel format, and Rec.709 color tags.
    """
    os.makedirs(os.path.dirname(before_proxy_path), exist_ok=True)
    os.makedirs(os.path.dirname(after_proxy_path), exist_ok=True)
    
    info = probe_video(source_path)
    fps = info.get("fps", 30.0)
    w = int(info.get("width", 1920))
    h = int(info.get("height", 1080))
    if h > max_height:
        target_h = max_height
        target_w = int(round(w * (max_height / h) / 2.0)) * 2
    else:
        target_h = h - (h % 2)
        target_w = w - (w % 2)
    target_w = max(2, target_w)
    target_h = max(2, target_h)
    scale_filter = f"scale={target_w}:{target_h}"
    formatted_lut_path = str(Path(lut_path).resolve().as_posix()).replace(':', '\\:')
    lut_filter = f"lut3d=file='{formatted_lut_path}',{scale_filter}"
    
    common_args = [
        '-r', str(fps),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-an'
    ]
    
    # 1. Render Before Proxy (Source without LUT)
    cmd_before = [
        get_ffmpeg_binary(),
        '-y',
        '-i', source_path,
        '-vf', scale_filter
    ] + common_args + [before_proxy_path]
    run_subprocess(cmd_before)
    
    # 2. Render After Proxy (Source with LUT applied)
    cmd_after = [
        get_ffmpeg_binary(),
        '-y',
        '-i', source_path,
        '-vf', lut_filter
    ] + common_args + [after_proxy_path]
    run_subprocess(cmd_after)
    
    return before_proxy_path, after_proxy_path

