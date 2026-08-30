import os
import sys
import argparse
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.media.ffmpeg import probe_video, extract_sampled_frames, apply_lut_and_render
from app.media.color import aggregate_shot_metrics, calculate_deterministic_match_params, compute_consistency_score
from app.media.lut import generate_3d_cube_lut

def run_local_match(ref_path: str, tgt_path: str, output_video: str = "output/matched.mp4", output_lut: str = "output/grade.cube"):
    print("=" * 70)
    print(" Autonomous Multimodal Colorist Assistant — Phase 1 Local Match")
    print("=" * 70)
    
    # 1. Probing media
    print(f"\n[1/5] Probing source videos...")
    ref_info = probe_video(ref_path)
    tgt_info = probe_video(tgt_path)
    print(f"  Reference: {ref_path} ({ref_info['width']}x{ref_info['height']}, {ref_info['fps']} fps, {ref_info['duration_sec']}s)")
    print(f"  Target:    {tgt_path} ({tgt_info['width']}x{tgt_info['height']}, {tgt_info['fps']} fps, {tgt_info['duration_sec']}s)")
    
    # 2. Extracting sampled keyframes
    print(f"\n[2/5] Extracting keyframe samples (25%, 50%, 75%)...")
    ref_frames, ref_times = extract_sampled_frames(ref_path)
    tgt_frames, tgt_times = extract_sampled_frames(tgt_path)
    
    # 3. Computing color statistics
    print(f"\n[3/5] Measuring numerical color metrics...")
    ref_metrics = aggregate_shot_metrics("ref", ref_path, ref_frames, ref_times, ref_info["fps"], ref_info["width"], ref_info["height"], ref_info["duration_sec"])
    tgt_metrics = aggregate_shot_metrics("tgt", tgt_path, tgt_frames, tgt_times, tgt_info["fps"], tgt_info["width"], tgt_info["height"], tgt_info["duration_sec"])
    
    print(f"  Reference Shot Metrics:")
    print(f"    - Avg Luminance: {ref_metrics.avg_luminance:.1f} / 255")
    print(f"    - CIELAB Mean:   L={ref_metrics.avg_lab_mean[0]:.1f}, a={ref_metrics.avg_lab_mean[1]:.1f}, b={ref_metrics.avg_lab_mean[2]:.1f}")
    print(f"    - Avg Chroma:    {ref_metrics.avg_chroma:.1f}")
    print(f"    - Dominant Cast: {ref_metrics.dominant_cast}")
    
    print(f"  Target Shot Metrics:")
    print(f"    - Avg Luminance: {tgt_metrics.avg_luminance:.1f} / 255")
    print(f"    - CIELAB Mean:   L={tgt_metrics.avg_lab_mean[0]:.1f}, a={tgt_metrics.avg_lab_mean[1]:.1f}, b={tgt_metrics.avg_lab_mean[2]:.1f}")
    print(f"    - Avg Chroma:    {tgt_metrics.avg_chroma:.1f}")
    print(f"    - Dominant Cast: {tgt_metrics.dominant_cast}")
    
    before_score = compute_consistency_score(ref_metrics, tgt_metrics)
    print(f"\n  -> Initial Consistency Score: {before_score.overall_score}/100 ({before_score.notes})")
    
    # 4. Deterministic grade calculation & 3D LUT generation
    print(f"\n[4/5] Computing statistical match & baking 3D LUT (33x33x33)...")
    params = calculate_deterministic_match_params(ref_metrics, tgt_metrics)
    print(f"  Calculated Parameters: L_gain={params.lab_l_gain}, L_offset={params.lab_l_offset}, a_gain={params.lab_a_gain}, b_gain={params.lab_b_gain}")
    
    lut_dir = os.path.dirname(output_lut)
    if lut_dir:
        os.makedirs(lut_dir, exist_ok=True)
    generate_3d_cube_lut(params, output_lut, size=33)
    print(f"  -> Saved 3D LUT: {output_lut}")
    
    # 5. FFmpeg render & evaluation
    print(f"\n[5/5] Rendering matched video with FFmpeg...")
    vid_dir = os.path.dirname(output_video)
    if vid_dir:
        os.makedirs(vid_dir, exist_ok=True)
    apply_lut_and_render(tgt_path, output_lut, output_video)
    print(f"  -> Rendered Output: {output_video}")
    
    # Post-grade verification
    out_frames, out_times = extract_sampled_frames(output_video)
    out_metrics = aggregate_shot_metrics("graded", output_video, out_frames, out_times, tgt_info["fps"], tgt_info["width"], tgt_info["height"], tgt_info["duration_sec"])
    after_score = compute_consistency_score(ref_metrics, out_metrics)
    
    print("\n" + "=" * 70)
    print(" MATCHING RESULTS SUMMARY")
    print("=" * 70)
    print(f" Metric                  | Before Grade       | After Grade")
    print(f" ------------------------+--------------------+--------------------")
    print(f" Overall Consistency     | {before_score.overall_score:5.1f} / 100        | {after_score.overall_score:5.1f} / 100")
    print(f" Luminance Similarity    | {before_score.luminance_similarity:5.1f} / 100        | {after_score.luminance_similarity:5.1f} / 100")
    print(f" Chroma Similarity       | {before_score.chroma_similarity:5.1f} / 100        | {after_score.chroma_similarity:5.1f} / 100")
    print(f" Color Dist. Similarity  | {before_score.color_distribution_similarity:5.1f} / 100        | {after_score.color_distribution_similarity:5.1f} / 100")
    print(f" Mean Luminance (Target) | {tgt_metrics.avg_luminance:5.1f}              | {out_metrics.avg_luminance:5.1f} (Ref: {ref_metrics.avg_luminance:.1f})")
    print("=" * 70)
    print(f"Success! Output video and .cube LUT ready.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Colorist Local Deterministic Match")
    parser.add_argument("--reference", default="tests/fixtures/sample_videos/neutral_reference.mp4", help="Reference video path")
    parser.add_argument("--target", default="tests/fixtures/sample_videos/underexposed.mp4", help="Target video path to match")
    parser.add_argument("--output", default="output/matched.mp4", help="Output video path")
    parser.add_argument("--lut", default="output/grade.cube", help="Output .cube LUT path")
    args = parser.parse_args()
    
    run_local_match(args.reference, args.target, args.output, args.lut)