import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from app.models.analysis import ShotSemanticAnalysis, ShotMetrics, CreativeSpecification
from app.models.grade import ColorGradeParams, GradePlan, ConsistencyScore, GradeResult
from app.tools.inspect_footage import inspect_all_shots_batched
from app.tools.measure_color import measure_shot_color
from app.tools.research import research_cinematography_principles, synthesize_creative_specification
from app.tools.calculate_grade import build_grade_plan, calculate_creative_grade
from app.tools.render import render_grade
from app.tools.evaluate import evaluate_grade
from app.media.color import is_log_profile, apply_color_grade_to_frame, compute_consistency_score
from app.media.lut import generate_3d_cube_lut, generate_shared_creative_look_lut
from app.media.ffmpeg import extract_sampled_frames, probe_video

class AutonomousColoristAgent:
    def __init__(self, work_dir: str = "output"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def process_sequence(
        self,
        video_paths: List[str],
        creative_prompt: str,
        reference_index: Optional[int] = None,
        color_profile: str = "auto",
        job_id: str = "job_default",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        events = []
        def log_event(msg: str):
            events.append(msg)
            print(f"[Agent] {msg}")
            if progress_callback:
                progress_callback(msg)
                
        log_event(f"Starting Autonomous Colorist job '{job_id}' with {len(video_paths)} clips.")
        
        # 1. PERCEIVE: Multimodal Video Inspection with Gemini
        log_event("Inspecting footage semantics with Gemini 3.5 Flash...")
        inspection_result = inspect_all_shots_batched(video_paths)
        semantic_analyses = inspection_result.shots
        is_continuous_sequence = (inspection_result.scene_relationship == "continuous_sequence")
        
        for s in semantic_analyses:
            log_event(f"  -> {s.shot_id}: {s.lighting_environment} ({s.time_of_day}), Exposure: {s.exposure_assessment}")
            
        # Determine Reference Shot
        if reference_index is not None and 0 <= reference_index < len(video_paths):
            ref_idx = reference_index
        else:
            rec_id = inspection_result.recommended_reference_shot_id
            ref_idx = 0
            for idx, s in enumerate(semantic_analyses):
                if s.shot_id == rec_id:
                    ref_idx = idx
                    break
                    
        ref_shot_id = f"shot_{chr(65 + ref_idx)}"
        ref_path = video_paths[ref_idx]
        ref_semantic = semantic_analyses[ref_idx]
        log_event(f"Selected '{ref_shot_id}' as master technical reference (Scene context: {ref_semantic.lighting_environment}).")
        
        # 2. MEASURE: Extract numerical color statistics in CIELAB space
        log_event("Measuring numerical color distributions in CIELAB space...")
        shot_metrics: List[ShotMetrics] = []
        is_log_flags: List[bool] = []
        cached_frames: List[List[np.ndarray]] = []
        cached_timestamps: List[List[float]] = []
        
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            metrics = measure_shot_color(path, shot_id=shot_id)
            shot_metrics.append(metrics)
            
            # Cache sampled frames for fast preview evaluation
            frames, timestamps = extract_sampled_frames(path, num_samples=6)
            cached_frames.append(frames)
            cached_timestamps.append(timestamps)
            
            # Check for Log profile
            shot_is_log = (
                color_profile == "Log" or
                color_profile == "dlog" or
                color_profile == "slog" or
                is_log_profile(metrics) or
                "log" in semantic_analyses[i].scene_description.lower()
            )
            is_log_flags.append(shot_is_log)
            if shot_is_log:
                log_event(f"  [CST] Log profile detected for {shot_id}. Color Space Transform (CST -> Rec.709) ACTIVE.")
            
        ref_metrics = shot_metrics[ref_idx]
        
        # 3. RESEARCH: Parallel Web Intelligence
        log_event(f"Researching cinematography principles on Parallel for: '{creative_prompt}'...")
        research_result = research_cinematography_principles(
            creative_prompt=creative_prompt,
            scene_context=ref_semantic.lighting_environment
        )
        if research_result.is_grounded and research_result.sources:
            log_event(f"Parallel retrieved {len(research_result.sources)} grounded cinematography references.")
        else:
            log_event("Parallel grounding unavailable — proceeding with ungrounded Gemini creative interpretation.")
        
        # Synthesize creative specification
        log_event("Synthesizing creative grading specification with Gemini...")
        creative_spec: CreativeSpecification = synthesize_creative_specification(
            creative_prompt=creative_prompt,
            research_result=research_result
        )
        log_event(f"Synthesized Look: '{creative_spec.look_title}' (Contrast: {creative_spec.contrast_intent}x, Saturation: {creative_spec.saturation_intent}x, Highlights: {creative_spec.highlight_bias}, Shadows: {creative_spec.shadow_bias})")
        
        # Export Shared Creative-Look LUT
        shared_lut_path = str(self.work_dir / f"{job_id}_shared_creative_look.cube")
        generate_shared_creative_look_lut(creative_spec, shared_lut_path)
        log_event(f"Exported Shared Creative-Look 3D LUT: {Path(shared_lut_path).name}")
        
        # 4. GRADE MASTER REFERENCE FIRST & ESTABLISH REFERENCE TARGET METRICS
        log_event(f"Grading Master Reference {ref_shot_id} with creative look '{creative_spec.look_title}'...")
        ref_plan = build_grade_plan(
            reference=ref_metrics,
            target=ref_metrics,
            target_semantic=ref_semantic,
            creative_spec=creative_spec,
            is_reference_shot=True,
            is_same_scene=False,
            is_log=is_log_flags[ref_idx]
        )
        
        # Grade reference sampled frames to establish true target metrics
        graded_ref_frames = [apply_color_grade_to_frame(f, ref_plan, is_log=is_log_flags[ref_idx]) for f in cached_frames[ref_idx]]
        graded_ref_metrics, _ = evaluate_grade(
            reference_metrics=ref_metrics,
            graded_video_or_frames=graded_ref_frames,
            evaluation_mode="same_scene_match",
            timestamps=cached_timestamps[ref_idx]
        )
        log_event(f"Established Graded Reference Target Metrics: Luminance={graded_ref_metrics.avg_luminance}, CIELAB={graded_ref_metrics.avg_lab_mean}")
        
        # 5. GRADE ALL SHOTS IN SEQUENCE WITH AUTONOMOUS EVALUATE-REVISE LOOP
        graded_results: List[GradeResult] = []
        
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            metrics = shot_metrics[i]
            semantic = semantic_analyses[i]
            shot_is_log = is_log_flags[i]
            is_ref = (i == ref_idx)
            is_same_scene = is_continuous_sequence or (
                not is_ref and
                semantic.lighting_environment == ref_semantic.lighting_environment and
                semantic.time_of_day == ref_semantic.time_of_day
            )
            eval_mode = "same_scene_match" if is_same_scene else "cross_scene_look_continuity"
            
            if is_ref:
                log_event(f"Finalizing Master Reference {shot_id}...")
                plan = ref_plan
                before_score = compute_consistency_score(graded_ref_metrics, metrics, evaluation_mode="same_scene_match")
                after_score = compute_consistency_score(graded_ref_metrics, graded_ref_metrics, evaluation_mode="same_scene_match")
                revisions_performed = 0
            else:
                if is_same_scene:
                    log_event(f"Matching {shot_id} to continuous scene reference {ref_shot_id}...")
                else:
                    log_event(f"Grading {shot_id} (Independent scene: {semantic.lighting_environment}, {semantic.time_of_day}) preserving scene exposure...")
                    
                before_score = compute_consistency_score(graded_ref_metrics, metrics, evaluation_mode=eval_mode)
                
                # Build initial plan
                plan = build_grade_plan(
                    reference=ref_metrics,
                    target=metrics,
                    target_semantic=semantic,
                    creative_spec=creative_spec,
                    is_reference_shot=False,
                    is_same_scene=is_same_scene,
                    is_log=shot_is_log
                )
                
                # AUTONOMOUS EVALUATE -> REVISE LOOP (Max 2 revisions)
                revisions_performed = 0
                max_revisions = 2
                
                for rev_idx in range(max_revisions + 1):
                    # Fast proxy evaluation on cached frames
                    graded_preview_frames = [apply_color_grade_to_frame(f, plan, is_log=shot_is_log) for f in cached_frames[i]]
                    eval_metrics, current_score = evaluate_grade(
                        reference_metrics=graded_ref_metrics,
                        graded_video_or_frames=graded_preview_frames,
                        evaluation_mode=eval_mode,
                        timestamps=cached_timestamps[i]
                    )
                    
                    if current_score.overall_score >= 75.0 or rev_idx == max_revisions:
                        after_score = current_score
                        log_event(f"  [Evaluate] {shot_id} evaluated: overall {after_score.overall_score}/100 (Chroma: {after_score.chromatic_similarity}, Tone: {after_score.tonal_similarity}, Clipping: {after_score.clipping_health}). ACCEPTED.")
                        break
                    else:
                        # Diagnostic Revision
                        revisions_performed += 1
                        diagnosis = current_score.diagnosis or "Discrepancy detected"
                        log_event(f"  [Evaluate] {shot_id} score {current_score.overall_score}/100 (Tone: {current_score.tonal_similarity}, Chroma: {current_score.chromatic_similarity}). Revision {revisions_performed}: {diagnosis}...")
                        
                        # Diagnose and adjust only relevant parameters
                        if current_score.tonal_similarity < 70.0 and is_same_scene:
                            p50_target = graded_ref_metrics.p50_luminance if graded_ref_metrics.p50_luminance > 0 else graded_ref_metrics.avg_luminance
                            p50_cand = eval_metrics.p50_luminance if eval_metrics.p50_luminance > 0 else eval_metrics.avg_luminance
                            ev_adj = float(np.log2(max(1.0, p50_target) / max(1.0, p50_cand)) * 0.45)
                            plan.technical_balance.exposure_ev += round(ev_adj, 2)
                            log_event(f"    -> Trimmed exposure by {ev_adj:+.2f} EV")
                            
                        if current_score.chromatic_similarity < 70.0:
                            delta_b = graded_ref_metrics.avg_lab_mean[2] - eval_metrics.avg_lab_mean[2]
                            delta_a = graded_ref_metrics.avg_lab_mean[1] - eval_metrics.avg_lab_mean[1]
                            plan.technical_balance.temperature += round(delta_b * 0.35, 1)
                            plan.technical_balance.tint += round(delta_a * 0.35, 1)
                            log_event(f"    -> Adjusted white balance (temp: {plan.technical_balance.temperature:+.1f}, tint: {plan.technical_balance.tint:+.1f})")
                            
                        if current_score.clipping_health < 80.0:
                            plan.creative_look.contrast = max(0.85, plan.creative_look.contrast * 0.92)
                            plan.output_transform.highlight_shoulder_threshold = 0.80
                            log_event(f"    -> Softened contrast curve to {round(plan.creative_look.contrast, 2)}x to protect highlights")
            
            # Render Final High-Quality Delivery Video & 3D LUT
            final_lut = str(self.work_dir / f"{job_id}_{shot_id}_grade.cube")
            final_video = str(self.work_dir / f"{job_id}_{shot_id}_graded.mp4")
            log_event(f"Rendering final master delivery video & 3D LUT for {shot_id}...")
            render_grade(path, plan, final_video, final_lut, is_preview=False, is_log=shot_is_log)
            
            # Construct comprehensive explanation
            log_desc = "Log CST normalization applied. " if shot_is_log else ""
            if is_ref:
                explanation = f"Master reference shot established the visual tone. {log_desc}Creative Look '{creative_spec.look_title}' applied with {creative_spec.contrast_intent}x contrast, {creative_spec.highlight_bias} highlights, and {creative_spec.shadow_bias} shadows."
            elif is_same_scene:
                rev_text = f" Harmonized over {revisions_performed} revision passes." if revisions_performed > 0 else ""
                explanation = f"Matched to continuous scene reference {ref_shot_id}. {log_desc}Harmonized chromatic balance (score: {after_score.chromatic_similarity}) and tonal exposure (score: {after_score.tonal_similarity}).{rev_text}"
            else:
                rev_text = f" Refined over {revisions_performed} revision passes." if revisions_performed > 0 else ""
                explanation = f"Independent scene ({semantic.lighting_environment}, {semantic.time_of_day}). {log_desc}Preserved natural scene exposure while applying the unified visual language and palette (continuity score: {after_score.overall_score}).{rev_text}"
                
            graded_results.append(GradeResult(
                reference_shot_id=ref_shot_id,
                target_shot_id=shot_id,
                plan=plan,
                params=plan.to_legacy_params(),
                lut_path=final_lut,
                shared_lut_path=shared_lut_path,
                output_video_path=final_video,
                before_consistency=before_score,
                after_consistency=after_score,
                revisions_performed=revisions_performed,
                explanation=explanation
            ))
            
        log_event(f"Autonomous color grading completed successfully for sequence of {len(video_paths)} shots.")
        
        return {
            "job_id": job_id,
            "reference_shot_id": ref_shot_id,
            "creative_specification": creative_spec.model_dump(),
            "research_citations": [c.model_dump() for c in creative_spec.citations],
            "shared_lut_path": shared_lut_path,
            "results": [r.model_dump() for r in graded_results],
            "events": events
        }