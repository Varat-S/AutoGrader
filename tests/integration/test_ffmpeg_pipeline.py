import os
import pytest
from pathlib import Path
from app.media.ffmpeg import probe_video, extract_sampled_frames, apply_lut_and_render, generate_proxy
from app.media.color import aggregate_shot_metrics, calculate_deterministic_match_params, compute_consistency_score
from app.media.lut import generate_3d_cube_lut

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent.parent / 'fixtures' / 'sample_videos'

def test_full_ffmpeg_match_pipeline(fixtures_dir, tmp_path):
    ref_path = str(fixtures_dir / 'neutral_reference.mp4')
    tgt_path = str(fixtures_dir / 'underexposed.mp4')
    
    assert os.path.exists(ref_path), 'Fixture neutral_reference.mp4 missing'
    assert os.path.exists(tgt_path), 'Fixture underexposed.mp4 missing'
    
    ref_info = probe_video(ref_path)
    tgt_info = probe_video(tgt_path)
    assert ref_info['width'] > 0
    assert tgt_info['duration_sec'] > 0
    
    ref_frames, ref_times = extract_sampled_frames(ref_path)
    tgt_frames, tgt_times = extract_sampled_frames(tgt_path)
    assert len(ref_frames) >= 3
    assert len(tgt_frames) >= 3
    
    ref_metrics = aggregate_shot_metrics('ref', ref_path, ref_frames, ref_times, ref_info['fps'], ref_info['width'], ref_info['height'], ref_info['duration_sec'])
    tgt_metrics = aggregate_shot_metrics('tgt', tgt_path, tgt_frames, tgt_times, tgt_info['fps'], tgt_info['width'], tgt_info['height'], tgt_info['duration_sec'])
    
    before_score = compute_consistency_score(ref_metrics, tgt_metrics)
    
    params = calculate_deterministic_match_params(ref_metrics, tgt_metrics)
    assert params.lab_l_offset > 0.0
    
    lut_path = str(tmp_path / 'match.cube')
    generate_3d_cube_lut(params, lut_path, size=33)
    assert os.path.exists(lut_path)
    
    output_video = str(tmp_path / 'graded_output.mp4')
    apply_lut_and_render(tgt_path, lut_path, output_video)
    assert os.path.exists(output_video)
    
    out_frames, out_times = extract_sampled_frames(output_video)
    out_metrics = aggregate_shot_metrics('graded', output_video, out_frames, out_times, tgt_info['fps'], tgt_info['width'], tgt_info['height'], tgt_info['duration_sec'])
    after_score = compute_consistency_score(ref_metrics, out_metrics)
    
    assert after_score.overall_score > before_score.overall_score
    print(f'\nConsistency score improved from {before_score.overall_score} to {after_score.overall_score}')
