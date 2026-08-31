import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.models.analysis import ShotSemanticAnalysis, ShotMetrics, CreativeSpecification
from app.models.grade import ColorGradeParams, ConsistencyScore, GradeResult
from app.tools.inspect_footage import inspect_all_shots_batched
from app.tools.measure_color import measure_shot_color
from app.tools.research import research_cinematography_principles, synthesize_creative_specification
from app.tools.calculate_grade import calculate_creative_grade
from app.tools.render import render_grade
from app.tools.evaluate import evaluate_grade
from app.media.color import is_log_profile

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
        
        # 1. PERCEIVE: Single Batched Multimodal Video Inspection with Gemini
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
        
        # 2. MEASURE: Extract numerical color statistics with OpenCV
        log_event("Measuring numerical color distributions in CIELAB space...")
        shot_metrics: List[ShotMetrics] = []
        is_log_flags: List[bool] = []
        
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            metrics = measure_shot_color(path, shot_id=shot_id)
            shot_metrics.append(metrics)
            
            # Check for Log profile (User selected or auto-detected by histogram & Gemini)
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
        log_event(f"Parallel retrieved {len(research_result.sources)} cinematography references.")
        
        # Synthesize creative specification
        log_event("Synthesizing creative grading specification with Gemini...")
        creative_spec: CreativeSpecification = synthesize_creative_specification(
            creative_prompt=creative_prompt,
            research_result=research_result
        )
        log_event(f"Synthesized Look: '{creative_spec.look_title}' (Contrast: {creative_spec.contrast_intent}x, Saturation: {creative_spec.saturation_intent}x)")
        
        # 4. ACT, EVALUATE & REVISE: Grade ALL Shots in Sequence
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
            
            if is_ref:
                log_event(f"Grading Master Reference {shot_id} with '{creative_spec.look_title}'...")
            elif is_same_scene:
                log_event(f"Matching {shot_id} to continuous scene reference {ref_shot_id}...")
            else:
                log_event(f"Grading {shot_id} (Independent scene: {semantic.lighting_environment}) with unified film look...")
                
            before_score = compute_consistency_score_helper(ref_metrics, metrics)
            
            # Skin protection check
            skin_prot = semantic.skin_protection_required or ref_semantic.skin_protection_required
            if skin_prot:
                log_event(f"Skin tone protection ACTIVE for {shot_id}.")
                
            params = calculate_creative_grade(
                reference=ref_metrics,
                target=metrics,
                target_semantic=semantic,
                creative_spec=creative_spec,
                is_reference_shot=is_ref,
                is_same_scene=is_same_scene
            )
            
            # Render Preview with Log CST
            preview_lut = str(self.work_dir / f"{job_id}_{shot_id}_preview.cube")
            preview_video = str(self.work_dir / f"{job_id}_{shot_id}_preview.mp4")
            log_event(f"Rendering preview for {shot_id}...")
            render_grade(path, params, preview_video, preview_lut, is_preview=True, is_log=shot_is_log)
            
            # Evaluate Preview
            eval_metrics, after_score = evaluate_grade(ref_metrics, preview_video)
            log_event(f"{shot_id} consistency: {after_score.overall_score}/100.")
            
            # Render Final Full-Quality Master with Log CST
            final_lut = str(self.work_dir / f"{job_id}_{shot_id}_grade.cube")
            final_video = str(self.work_dir / f"{job_id}_{shot_id}_graded.mp4")
            log_event(f"Rendering final master video & 3D LUT for {shot_id}...")
            render_grade(path, params, final_video, final_lut, is_preview=False, is_log=shot_is_log)
            
            log_desc = "Log -> Rec.709 CST applied. " if shot_is_log else ""
            if is_ref:
                explanation = (
                    f"Master technical reference. {log_desc}Applied creative '{creative_spec.look_title}' film stock emulation "
                    f"(Contrast: {creative_spec.contrast_intent}x, Saturation: {creative_spec.saturation_intent}x). "
                    f"{'Preserved natural skin tones. ' if skin_prot else ''}"
                )
            elif is_same_scene:
                explanation = (
                    f"Continuous scene match to {ref_shot_id}. {log_desc}Applied unified '{creative_spec.look_title}' style. "
                    f"{'Protected skin tones. ' if skin_prot else ''}"
                    f"Consistency score: {after_score.overall_score}/100."
                )
            else:
                explanation = (
                    f"Scene-aware film emulation ({semantic.lighting_environment}). {log_desc}Preserved natural scene exposure "
                    f"while harmonizing color palette and applying '{creative_spec.look_title}' style. "
                    f"{'Protected skin tones. ' if skin_prot else ''}"
                )
                
            graded_results.append(GradeResult(
                reference_shot_id=ref_shot_id,
                target_shot_id=shot_id,
                params=params,
                lut_path=final_lut,
                output_video_path=final_video,
                before_consistency=before_score,
                after_consistency=after_score,
                explanation=explanation
            ))
            
        log_event("Workflow complete! All shots successfully graded.")
        
        return {
            "job_id": job_id,
            "reference_shot_id": ref_shot_id,
            "creative_specification": creative_spec.model_dump(),
            "research_citations": [c.model_dump() for c in creative_spec.citations],
            "results": [r.model_dump() for r in graded_results],
            "events": events
        }

def compute_consistency_score_helper(a: ShotMetrics, b: ShotMetrics) -> ConsistencyScore:
    from app.media.color import compute_consistency_score
    return compute_consistency_score(a, b)