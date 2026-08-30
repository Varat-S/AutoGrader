import os
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_scene(width=640, height=360, frame_count=60, fps=30, variant='neutral') -> str:
    fixtures_dir = Path(__file__).parent / 'sample_videos'
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = str(fixtures_dir / f'{variant}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    
    for i in range(frame_count):
        # Base canvas: sky gradient (top) and ground (bottom)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky: blue-ish gradient
        for y in range(height // 2):
            val = 200 - int(y * 60 / (height // 2))
            frame[y, :] = [val, int(val * 0.85), int(val * 0.6)] # BGR
            
        # Ground: earthy/greenish
        for y in range(height // 2, height):
            progress = (y - height // 2) / (height // 2)
            frame[y, :] = [40, int(90 + progress * 30), int(60 + progress * 20)]
            
        # Foreground subject (simulated person/skin tones in center circle)
        # Moving slightly across frames
        center_x = int(width // 2 + np.sin(i * 0.1) * 20)
        center_y = int(height // 2 + np.cos(i * 0.1) * 10)
        
        # Skin tone color: roughly RGB(210, 160, 130) -> BGR(130, 160, 210)
        cv2.circle(frame, (center_x, center_y), 60, (130, 160, 210), -1)
        
        # Neutral gray card in corner (to test white balance)
        cv2.rectangle(frame, (30, height - 70), (90, height - 30), (128, 128, 128), -1)
        
        # White highlight patch
        cv2.rectangle(frame, (100, height - 70), (140, height - 30), (240, 240, 240), -1)
        
        # Deep shadow patch
        cv2.rectangle(frame, (150, height - 70), (190, height - 30), (15, 15, 15), -1)
        
        # Apply synthetic distortion for variants
        if variant == 'underexposed':
            # Darken by ~1.5 stops
            frame = (frame.astype(np.float32) * 0.35).astype(np.uint8)
        elif variant == 'warm_cast':
            # Heavy tungsten/warm orange cast (boost red, reduce blue)
            f_float = frame.astype(np.float32)
            f_float[:, :, 2] = np.clip(f_float[:, :, 2] * 1.45, 0, 255) # R
            f_float[:, :, 0] = np.clip(f_float[:, :, 0] * 0.60, 0, 255) # B
            frame = f_float.astype(np.uint8)
        elif variant == 'oversaturated':
            # High chroma
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        out.write(frame)
        
    out.release()
    print(f'Generated fixture: {out_file}')
    return out_file

if __name__ == '__main__':
    for v in ['neutral_reference', 'underexposed', 'warm_cast', 'oversaturated']:
        create_synthetic_scene(variant=v)
