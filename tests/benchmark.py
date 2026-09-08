#!/usr/bin/env python3
"""
Deterministic Benchmark Runner for AutoGrader
Runs the full color-grading workflow on local offline test fixtures without external API calls.
Outputs benchmark_results.json with commit SHA, execution timing, consistency metrics, and status.
"""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import time
import subprocess
from datetime import datetime, timezone
from unittest.mock import patch

from app.agent import AutonomousColoristAgent
from app.models.grade import InputProfile, ShotProfileSelection
from app.models.analysis import (
    SequenceInspectionResult,
    ShotSemanticAnalysis,
    CinematographyResearchResult,
    CreativeSpecification,
    SearchCitation,
)

def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception as e:
        return f"unknown ({e})"

def run_benchmark(output_json_path: str = "benchmark_results.json") -> dict:
    repo_root = Path(__file__).parent.parent
    fixtures_dir = repo_root / "tests" / "fixtures" / "sample_videos"
    
    video_paths = [
        str(fixtures_dir / "neutral_reference.mp4"),
        str(fixtures_dir / "underexposed.mp4"),
        str(fixtures_dir / "warm_cast.mp4"),
    ]
    
    for v in video_paths:
        if not os.path.exists(v):
            raise FileNotFoundError(f"Required benchmark fixture not found: {v}")

    work_dir = repo_root / "output" / "benchmark_run"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Realistic offline inspection result
    mock_inspection = SequenceInspectionResult(
        shots=[
            ShotSemanticAnalysis(
                shot_id="shot_A",
                scene_group_id="group_1",
                relationship_to_reference="reference",
                scene_description="Daylight reference shot",
                lighting_environment="outdoor daylight",
                time_of_day="day",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.95,
            ),
            ShotSemanticAnalysis(
                shot_id="shot_B",
                scene_group_id="group_1",
                relationship_to_reference="same_scene",
                scene_description="Daylight underexposed shot",
                lighting_environment="outdoor daylight",
                time_of_day="day",
                exposure_assessment="underexposed",
                target_exposure_compensation_ev=1.2,
                black_point_lift=2.0,
                people_present=False,
                dominant_color_cast="neutral",
                reference_suitability_score=0.60,
            ),
            ShotSemanticAnalysis(
                shot_id="shot_C",
                scene_group_id="group_2",
                relationship_to_reference="independent_scene",
                scene_description="Warm cast independent scene",
                lighting_environment="golden hour",
                time_of_day="golden_hour",
                exposure_assessment="balanced",
                target_exposure_compensation_ev=0.0,
                black_point_lift=2.5,
                people_present=False,
                dominant_color_cast="warm / golden",
                reference_suitability_score=0.75,
            ),
        ],
        recommended_reference_shot_id="shot_A",
        scene_relationship="mixed_sequence",
    )

    mock_research = CinematographyResearchResult(
        query="desert sci-fi",
        objective="research",
        sources=[SearchCitation(title="American Cinematographer", url="https://theasc.com/article", excerpt="Warm golden highlights, cool slate shadows.")],
        is_grounded=True,
    )

    mock_spec = CreativeSpecification(
        look_title="Desert Sci-Fi",
        target_aesthetic="Warm golden highlights, muted saturation, cool slate shadows",
        contrast_intent=1.12,
        saturation_intent=1.05,
        highlight_bias="warm golden",
        shadow_bias="cool slate",
        black_level_treatment="filmic lifted",
        temperature_shift=3.0,
        tint_shift=-1.0,
        black_mist_diffusion_strength=0.2,
        cinematography_principles=["Highlight roll-off", "Complementary color separation"],
        citations=mock_research.sources,
    )

    agent = AutonomousColoristAgent(work_dir=str(work_dir))

    # Profiles explicitly confirmed for each test clip
    input_profiles = [
        ShotProfileSelection(shot_index=0, profile=InputProfile.REC709, user_confirmed=True),
        ShotProfileSelection(shot_index=1, profile=InputProfile.REC709, user_confirmed=True),
        ShotProfileSelection(shot_index=2, profile=InputProfile.REC709, user_confirmed=True),
    ]

    commit_sha = get_git_commit_sha()
    start_time = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()

    with patch("app.agent.inspect_all_shots_batched", return_value=mock_inspection), \
         patch("app.agent.research_cinematography_principles", return_value=mock_research), \
         patch("app.agent.synthesize_creative_specification", return_value=mock_spec):

        result = agent.process_sequence(
            video_paths=video_paths,
            creative_prompt="Restrained desert sci-fi aesthetic.",
            input_profiles=input_profiles,
            job_id="benchmark",
        )

    total_duration_sec = round(time.perf_counter() - start_time, 3)

    # Process per-shot results
    shots_summary = []
    overall_scores = []
    lut_validity = []
    video_validity = []

    for r in result["results"]:
        target_id = r["target_shot_id"]
        after_score = r.get("after_consistency", {})
        lut_path = r.get("lut_path", "")
        vid_path = r.get("output_video_path", "")

        lut_ok = os.path.exists(lut_path) and os.path.getsize(lut_path) > 100
        vid_ok = os.path.exists(vid_path) and os.path.getsize(vid_path) > 1000
        lut_validity.append(lut_ok)
        video_validity.append(vid_ok)

        score_val = after_score.get("overall_score", 0.0)
        overall_scores.append(score_val)

        shots_summary.append({
            "shot_id": target_id,
            "state": r.get("state"),
            "revisions_performed": r.get("revisions_performed", 0),
            "input_profile": r.get("plan", {}).get("input_transform", {}).get("profile"),
            "consistency_score": {
                "overall": score_val,
                "tonal": after_score.get("tonal_similarity", 0.0),
                "chromatic": after_score.get("chromatic_similarity", 0.0),
                "clipping_health": after_score.get("clipping_health", 0.0),
            },
            "lut_valid": lut_ok,
            "video_valid": vid_ok,
        })

    shared_lut_ok = os.path.exists(result["shared_lut_path"]) and os.path.getsize(result["shared_lut_path"]) > 100

    seq_avg_score = round(sum(overall_scores) / max(1, len(overall_scores)), 2)
    passed_threshold = (seq_avg_score >= 75.0) and all(lut_validity) and all(video_validity) and shared_lut_ok

    benchmark_data = {
        "git_commit_sha": commit_sha,
        "timestamp_utc": start_iso,
        "status": "PASS" if passed_threshold else "FAIL",
        "total_duration_sec": total_duration_sec,
        "sequence_average_score": seq_avg_score,
        "pass_threshold": 75.0,
        "reference_shot_id": result.get("reference_shot_id"),
        "shared_lut_valid": shared_lut_ok,
        "shots": shots_summary,
        "normalization_health": [
            {
                "shot_id": n.get("shot_id"),
                "state": n.get("state"),
                "passed": n.get("passed"),
            }
            for n in result.get("normalization_results", [])
        ],
    }

    out_file = Path(output_json_path)
    if not out_file.is_absolute():
        out_file = repo_root / out_file

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n[Benchmark] Status: {benchmark_data['status']}")
    print(f"[Benchmark] Git Commit: {commit_sha}")
    print(f"[Benchmark] Total Duration: {total_duration_sec}s")
    print(f"[Benchmark] Sequence Average Score: {seq_avg_score}/100")
    for s in shots_summary:
        print(f"  - {s['shot_id']}: State={s['state']}, Revisions={s['revisions_performed']}, Score={s['consistency_score']['overall']} (Tone: {s['consistency_score']['tonal']}, Chroma: {s['consistency_score']['chromatic']})")
    print(f"[Benchmark] Wrote results to {out_file}")

    return benchmark_data

if __name__ == "__main__":
    data = run_benchmark()
    if data["status"] != "PASS":
        sys.exit(1)
    sys.exit(0)
