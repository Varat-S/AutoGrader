import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from app.models.analysis import ShotSemanticAnalysis, ShotMetrics, CreativeSpecification
from app.models.grade import (
    ColorGradeParams,
    GradePlan,
    ConsistencyScore,
    GradeResult,
    RevisionRecord
)
from app.tools.inspect_footage import inspect_all_shots_batched
from app.tools.measure_color import measure_shot_color
from app.tools.research import research_cinematography_principles, synthesize_creative_specification
from app.tools.calculate_grade import build_grade_plan
from app.tools.render import render_grade
from app.tools.evaluate import evaluate_grade
from app.media.color import (
    is_log_profile,
    apply_color_grade_to_frame,
    compute_consistency_score,
    calculate_deterministic_match_params,
    aggregate_shot_metrics
)
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
        log_event("Inspecting footage semantics with Gemini...")
        inspection_result = inspect_all_shots_batched(video_paths)
        semantic_analyses = inspection_result.shots
        
        for s in semantic_analyses:
            log_event(f"  -> {s.shot_id} [{s.scene_group_id}]: {s.lighting_environment} ({s.time_of_day}), Exposure: {s.exposure_assessment}, Rel: {s.relationship_to_reference}")
            
        # Determine Master Reference Shot
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
        ref_semantic.relationship_to_reference = "reference"
        log_event(f"Selected '{ref_shot_id}' as master technical reference (Group: {ref_semantic.scene_group_id}, Context: {ref_semantic.lighting_environment}).")
        
        # 2. MEASURE: Extract numerical color statistics & Sample Keyframes
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
            
            # Authoritative Input Profile Precedence
            if color_profile == "Rec.709":
                shot_is_log = False
            elif color_profile in ["Log", "Generic Log"]:
                shot_is_log = True
            else:
                # Auto
                shot_is_log = bool(is_log_profile(metrics) or "log" in semantic_analyses[i].scene_description.lower())
                
            is_log_flags.append(shot_is_log)
            if shot_is_log:
                log_event(f"  [Input Transform] Generic Flat/Log normalization ACTIVE for {shot_id}.")
            
        ref_metrics = shot_metrics[ref_idx]
        
        # 3. RESEARCH: Parallel Web Intelligence & Gemini Look Synthesis
        log_event(f"Researching cinematography principles on Parallel for: '{creative_prompt}'...")
        research_result = research_cinematography_principles(
            creative_prompt=creative_prompt,
            scene_context=ref_semantic.lighting_environment
        )
        if research_result.is_grounded and research_result.sources:
            log_event(f"Parallel retrieved {len(research_result.sources)} grounded cinematography references.")
        else:
            log_event("Parallel grounding unavailable — proceeding with ungrounded Gemini creative interpretation.")
            
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
            color_profile=color_profile
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
        
        # Measure intermediate balanced reference frames for same-scene matching in active space
        ref_balanced_frames = [apply_color_grade_to_frame(f, GradePlan(
            shot_id=ref_shot_id,
            input_transform=ref_plan.input_transform,
            technical_balance=ref_plan.technical_balance
        ), is_log=is_log_flags[ref_idx]) for f in cached_frames[ref_idx]]
        
        balanced_ref_metrics = aggregate_shot_metrics(
            shot_id=f"{ref_shot_id}_balanced",
            video_path="",
            frames=ref_balanced_frames,
            timestamps=cached_timestamps[ref_idx],
            fps=ref_metrics.fps,
            width=ref_metrics.width,
            height=ref_metrics.height,
            duration_sec=ref_metrics.duration_sec
        )
        
        # 5. GRADE ALL SHOTS IN SEQUENCE WITH HONEST REVISION STATE MACHINE
        graded_results: List[GradeResult] = []
        
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            metrics = shot_metrics[i]
            semantic = semantic_analyses[i]
            shot_is_log = is_log_flags[i]
            is_ref = (i == ref_idx)
            
            # Explicit Per-Shot Scene Grouping
            is_same_scene = not is_ref and (
                semantic.relationship_to_reference == "same_scene" or
                semantic.scene_group_id == ref_semantic.scene_group_id
            )
            eval_mode = "same_scene_match" if is_same_scene else "cross_scene_look_continuity"
            
            if is_ref:
                log_event(f"Finalizing Master Reference {shot_id}...")
                plan = ref_plan
                before_score = compute_consistency_score(graded_ref_metrics, metrics, evaluation_mode="same_scene_match")
                after_score = compute_consistency_score(graded_ref_metrics, graded_ref_metrics, evaluation_mode="same_scene_match")
                revisions_performed = 0
                final_state = "ACCEPTED"
                history = [RevisionRecord(
                    iteration=0,
                    state="ACCEPTED",
                    action_taken="Master technical reference established standard.",
                    overall_score_before=after_score.overall_score,
                    overall_score_after=after_score.overall_score,
                    diagnosis="Reference baseline",
                    parameter_deltas={}
                )]
                best_plan = plan
            else:
                if is_same_scene:
                    log_event(f"Matching {shot_id} to continuous scene reference {ref_shot_id} (Group: {semantic.scene_group_id})...")
                    # Calculate matching in post-normalization/balanced intermediate space
                    cand_balanced_frames = [apply_color_grade_to_frame(f, GradePlan(
                        shot_id=shot_id,
                        input_transform=ref_plan.input_transform,
                        technical_balance=ref_plan.technical_balance
                    ), is_log=shot_is_log) for f in cached_frames[i]]
                    
                    balanced_cand_metrics = aggregate_shot_metrics(
                        shot_id=f"{shot_id}_balanced",
                        video_path="",
                        frames=cand_balanced_frames,
                        timestamps=cached_timestamps[i],
                        fps=metrics.fps,
                        width=metrics.width,
                        height=metrics.height,
                        duration_sec=metrics.duration_sec
                    )
                    
                    matched_params = calculate_deterministic_match_params(balanced_ref_metrics, balanced_cand_metrics, strength=0.90)
                else:
                    log_event(f"Grading {shot_id} (Independent scene group: {semantic.scene_group_id}, {semantic.lighting_environment}) preserving scene exposure...")
                    matched_params = None
                    
                before_score = compute_consistency_score(
                    reference=graded_ref_metrics,
                    candidate=metrics,
                    evaluation_mode=eval_mode,
                    ref_plan=ref_plan,
                    cand_plan=None
                )
                
                # Build initial plan
                plan = build_grade_plan(
                    reference=ref_metrics,
                    target=metrics,
                    target_semantic=semantic,
                    creative_spec=creative_spec,
                    is_reference_shot=False,
                    is_same_scene=is_same_scene,
                    color_profile=color_profile,
                    matched_params=matched_params
                )
                
                # REVISION STATE MACHINE
                # Initial preview evaluation
                initial_preview_frames = [apply_color_grade_to_frame(f, plan, is_log=shot_is_log) for f in cached_frames[i]]
                eval_metrics, initial_score = evaluate_grade(
                    reference_metrics=graded_ref_metrics,
                    graded_video_or_frames=initial_preview_frames,
                    evaluation_mode=eval_mode,
                    timestamps=cached_timestamps[i],
                    ref_plan=ref_plan,
                    cand_plan=plan
                )
                
                best_plan = plan.model_copy(deep=True)
                best_score = initial_score
                history: List[RevisionRecord] = []
                revisions_performed = 0
                max_revisions = 2
                
                log_event(f"  [Evaluate] Initial grade for {shot_id}: overall {initial_score.overall_score}/100 (Tone: {initial_score.tonal_similarity}, Chroma: {initial_score.chromatic_similarity}, Clip: {initial_score.clipping_health})")
                
                if initial_score.overall_score >= 75.0:
                    final_state = "ACCEPTED"
                    history.append(RevisionRecord(
                        iteration=0,
                        state="ACCEPTED",
                        action_taken="Initial grade plan satisfies consistency tolerance.",
                        overall_score_before=initial_score.overall_score,
                        overall_score_after=initial_score.overall_score,
                        diagnosis=initial_score.diagnosis or "Pass",
                        parameter_deltas={}
                    ))
                    log_event(f"  [State] {shot_id} -> ACCEPTED on initial evaluation.")
                else:
                    final_state = "MAX_REVISIONS_REACHED"
                    history.append(RevisionRecord(
                        iteration=0,
                        state="INITIAL_EVALUATION",
                        action_taken=f"Initial score below threshold ({initial_score.overall_score} < 75.0). Beginning diagnostic revisions.",
                        overall_score_before=initial_score.overall_score,
                        overall_score_after=initial_score.overall_score,
                        diagnosis=initial_score.diagnosis or "Discrepancy detected",
                        parameter_deltas={}
                    ))
                    
                    for rev_idx in range(1, max_revisions + 1):
                        revisions_performed += 1
                        proposed_plan = best_plan.model_copy(deep=True)
                        deltas = {}
                        action_desc = ""
                        
                        if is_same_scene:
                            # Same-Scene Diagnostic Policy
                            if best_score.tonal_similarity < 70.0:
                                p50_target = graded_ref_metrics.p50_luminance if graded_ref_metrics.p50_luminance > 0 else graded_ref_metrics.avg_luminance
                                p50_cand = eval_metrics.p50_luminance if eval_metrics.p50_luminance > 0 else eval_metrics.avg_luminance
                                ev_adj = float(np.clip(np.log2(max(1.0, p50_target) / max(1.0, p50_cand)) * 0.45, -1.0, 1.0))
                                if abs(ev_adj) > 0.05:
                                    new_ev = float(np.clip(proposed_plan.technical_balance.exposure_ev + ev_adj, -2.5, 2.5))
                                    deltas["exposure_ev"] = round(new_ev - proposed_plan.technical_balance.exposure_ev, 2)
                                    proposed_plan.technical_balance.exposure_ev = round(new_ev, 2)
                                    action_desc = f"Adjusted exposure by {deltas['exposure_ev']:+.2f} EV"
                                    
                            elif best_score.chromatic_similarity < 70.0:
                                delta_b = graded_ref_metrics.avg_lab_mean[2] - eval_metrics.avg_lab_mean[2]
                                delta_a = graded_ref_metrics.avg_lab_mean[1] - eval_metrics.avg_lab_mean[1]
                                t_adj = float(np.clip(delta_b * 0.35, -15.0, 15.0))
                                tint_adj = float(np.clip(delta_a * 0.35, -10.0, 10.0))
                                if abs(t_adj) > 0.5 or abs(tint_adj) > 0.5:
                                    new_temp = float(np.clip(proposed_plan.technical_balance.temperature + t_adj, -40.0, 40.0))
                                    new_tint = float(np.clip(proposed_plan.technical_balance.tint + tint_adj, -25.0, 25.0))
                                    deltas["temperature"] = round(new_temp - proposed_plan.technical_balance.temperature, 1)
                                    deltas["tint"] = round(new_tint - proposed_plan.technical_balance.tint, 1)
                                    proposed_plan.technical_balance.temperature = round(new_temp, 1)
                                    proposed_plan.technical_balance.tint = round(new_tint, 1)
                                    action_desc = f"Refined white balance (temp: {deltas['temperature']:+.1f}, tint: {deltas['tint']:+.1f})"
                                    
                            elif best_score.clipping_health < 80.0:
                                new_contrast = max(0.80, round(proposed_plan.creative_look.contrast * 0.90, 2))
                                if abs(new_contrast - proposed_plan.creative_look.contrast) > 0.02:
                                    deltas["contrast"] = round(new_contrast - proposed_plan.creative_look.contrast, 2)
                                    proposed_plan.creative_look.contrast = new_contrast
                                    action_desc = f"Softened contrast curve to {new_contrast}x"
                        else:
                            # Cross-Scene Diagnostic Policy (Content-Independent Probe Response)
                            if best_score.clipping_health < 80.0:
                                new_contrast = max(0.80, round(proposed_plan.creative_look.contrast * 0.90, 2))
                                if abs(new_contrast - proposed_plan.creative_look.contrast) > 0.02:
                                    deltas["contrast"] = round(new_contrast - proposed_plan.creative_look.contrast, 2)
                                    proposed_plan.creative_look.contrast = new_contrast
                                    action_desc = f"Softened contrast curve to {new_contrast}x to eliminate clipping"
                            elif best_score.chromatic_similarity < 70.0:
                                # Probe split-tone bias mismatch
                                proposed_plan.creative_look.highlight_rgb_offset = ref_plan.creative_look.highlight_rgb_offset.copy()
                                proposed_plan.creative_look.shadow_rgb_offset = ref_plan.creative_look.shadow_rgb_offset.copy()
                                deltas["look_biases"] = "realigned_to_creative_spec"
                                action_desc = "Realigned highlight/shadow split biases with creative specification"
                                
                        if not deltas:
                            history.append(RevisionRecord(
                                iteration=rev_idx,
                                state="NO_ACTIONABLE_REVISION",
                                action_taken="No further actionable parameter adjustment diagnosed.",
                                overall_score_before=best_score.overall_score,
                                overall_score_after=best_score.overall_score,
                                diagnosis=best_score.diagnosis or "Unchanged",
                                parameter_deltas={}
                            ))
                            log_event(f"  [State] {shot_id} -> NO_ACTIONABLE_REVISION. Retaining best plan.")
                            break
                            
                        # Evaluate proposed plan
                        log_event(f"  [Revision {rev_idx} PROPOSED] {action_desc}...")
                        prop_preview_frames = [apply_color_grade_to_frame(f, proposed_plan, is_log=shot_is_log) for f in cached_frames[i]]
                        prop_metrics, prop_score = evaluate_grade(
                            reference_metrics=graded_ref_metrics,
                            graded_video_or_frames=prop_preview_frames,
                            evaluation_mode=eval_mode,
                            timestamps=cached_timestamps[i],
                            ref_plan=ref_plan,
                            cand_plan=proposed_plan
                        )
                        
                        if prop_score.overall_score > best_score.overall_score + 0.5:
                            # Proposal improved the objective
                            history.append(RevisionRecord(
                                iteration=rev_idx,
                                state="REVISION_IMPROVED",
                                action_taken=f"{action_desc} (improved score: {best_score.overall_score} -> {prop_score.overall_score})",
                                overall_score_before=best_score.overall_score,
                                overall_score_after=prop_score.overall_score,
                                diagnosis=prop_score.diagnosis or "Improved",
                                parameter_deltas=deltas
                            ))
                            best_plan = proposed_plan.model_copy(deep=True)
                            best_score = prop_score
                            eval_metrics = prop_metrics
                            log_event(f"  [State] Revision {rev_idx} IMPROVED score to {best_score.overall_score}/100. Updated best plan.")
                            
                            if best_score.overall_score >= 75.0:
                                final_state = "ACCEPTED"
                                log_event(f"  [State] {shot_id} -> ACCEPTED.")
                                break
                        else:
                            # Proposal failed to improve or degraded score -> revert
                            history.append(RevisionRecord(
                                iteration=rev_idx,
                                state="REVISION_REJECTED",
                                action_taken=f"{action_desc} rejected (score did not improve: {prop_score.overall_score} vs best {best_score.overall_score})",
                                overall_score_before=best_score.overall_score,
                                overall_score_after=prop_score.overall_score,
                                diagnosis=prop_score.diagnosis or "Reverted",
                                parameter_deltas=deltas
                            ))
                            log_event(f"  [State] Revision {rev_idx} REJECTED ({prop_score.overall_score} vs best {best_score.overall_score}). Reverted to best plan.")
                            
                    if final_state == "MAX_REVISIONS_REACHED":
                        log_event(f"  [State] {shot_id} -> MAX_REVISIONS_REACHED (final score: {best_score.overall_score}/100). Rendering verified best plan.")
                        
                after_score = best_score
                plan = best_plan
            
            # Render Final High-Quality Delivery Video & 3D LUT using best_plan
            final_lut = str(self.work_dir / f"{job_id}_{shot_id}_grade.cube")
            final_video = str(self.work_dir / f"{job_id}_{shot_id}_graded.mp4")
            log_event(f"Rendering final master delivery video & 3D LUT for {shot_id} (State: {final_state})...")
            render_grade(path, best_plan, final_video, final_lut, is_preview=False, is_log=shot_is_log)
            
            # Construct honest explanation
            log_desc = "Log normalization applied. " if shot_is_log else ""
            if is_ref:
                explanation = f"Master technical reference shot established standard. {log_desc}Creative Look '{creative_spec.look_title}' applied ({creative_spec.contrast_intent}x contrast, {creative_spec.highlight_bias} highlights, {creative_spec.shadow_bias} shadows)."
            elif is_same_scene:
                explanation = f"Matched to continuous scene reference {ref_shot_id} (Group {semantic.scene_group_id}). {log_desc}Harmonized chromatic balance (score: {after_score.chromatic_similarity}) and tonal exposure (score: {after_score.tonal_similarity}). State: {final_state} over {revisions_performed} revisions."
            else:
                explanation = f"Independent scene group {semantic.scene_group_id} ({semantic.lighting_environment}, {semantic.time_of_day}). {log_desc}Preserved natural scene exposure and shadow depth while aligning creative look transform invariants (continuity score: {after_score.overall_score}). State: {final_state}."
                
            graded_results.append(GradeResult(
                reference_shot_id=ref_shot_id,
                target_shot_id=shot_id,
                state=final_state,
                plan=best_plan,
                params=best_plan.to_legacy_params(),
                lut_path=final_lut,
                shared_lut_path=shared_lut_path,
                output_video_path=final_video,
                before_consistency=before_score,
                after_consistency=after_score,
                revisions_performed=revisions_performed,
                history=history,
                explanation=explanation
            ))
            
        log_event(f"Autonomous color grading completed for sequence of {len(video_paths)} shots.")
        
        return {
            "job_id": job_id,
            "reference_shot_id": ref_shot_id,
            "creative_specification": creative_spec.model_dump(),
            "research_citations": [c.model_dump() for c in creative_spec.citations],
            "shared_lut_path": shared_lut_path,
            "results": [r.model_dump() for r in graded_results],
            "events": events
        }