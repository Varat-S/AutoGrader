import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.models.analysis import ShotSemanticAnalysis, ShotMetrics, CreativeSpecification
from app.models.grade import ColorGradeParams, ConsistencyScore, GradeResult
from app.tools.inspect_footage import inspect_footage_semantics
from app.tools.measure_color import measure_shot_color
from app.tools.research import research_cinematography_principles, synthesize_creative_specification
from app.tools.calculate_grade import calculate_creative_grade
from app.tools.render import render_grade
from app.tools.evaluate import evaluate_grade

class AutonomousColoristAgent:
    def __init__(self, work_dir: str = "output"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def process_sequence(
        self,
        video_paths: List[str],
        creative_prompt: str,
        reference_index: Optional[int] = None,
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
        log_event("Inspecting footage semantics with Gemini 3.6 Flash...")
        semantic_analyses: List[ShotSemanticAnalysis] = []
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            log_event(f"Analyzing {shot_id} semantics ({Path(path).name})...")
            analysis = inspect_footage_semantics(path, shot_id=shot_id)
            semantic_analyses.append(analysis)
            log_event(f"  -> {shot_id}: {analysis.lighting_environment} ({analysis.time_of_day}), Exposure: {analysis.exposure_assessment}")
            
        # Determine Reference Shot
        if reference_index is not None and 0 <= reference_index < len(video_paths):
            ref_idx = reference_index
        else:
            ref_idx = max(range(len(semantic_analyses)), key=lambda i: semantic_analyses[i].reference_suitability_score)
            
        ref_shot_id = f"shot_{chr(65 + ref_idx)}"
        ref_path = video_paths[ref_idx]
        ref_semantic = semantic_analyses[ref_idx]
        log_event(f"Selected '{ref_shot_id}' as master technical reference (Suitability: {ref_semantic.reference_suitability_score:.2f}).")
        
        # 2. MEASURE: Extract numerical color statistics with OpenCV
        log_event("Measuring numerical color distributions in CIELAB space...")
        shot_metrics: List[ShotMetrics] = []
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            metrics = measure_shot_color(path, shot_id=shot_id)
            shot_metrics.append(metrics)
            
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
            is_ref = (i == ref_idx)
            
            # Determine if target shot is from same continuous scene as reference
            is_same_scene = (
                not is_ref and
                semantic.lighting_environment == ref_semantic.lighting_environment and
                semantic.time_of_day == ref_semantic.time_of_day
            )
            
            if is_ref:
                log_event(f"Grading Master Reference {shot_id} with '{creative_spec.look_title}'...")
            elif is_same_scene:
                log_event(f"Matching {shot_id} to continuous scene reference {ref_shot_id}...")
            else:
                log_event(f"Grading {shot_id} (Independent scene: {semantic.lighting_environment}) with unified film emulation...")
                
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
            
            # Render Preview
            preview_lut = str(self.work_dir / f"{job_id}_{shot_id}_preview.cube")
            preview_video = str(self.work_dir / f"{job_id}_{shot_id}_preview.mp4")
            log_event(f"Rendering preview for {shot_id}...")
            render_grade(path, params, preview_video, preview_lut, is_preview=True)
            
            # Evaluate Preview
            eval_metrics, after_score = evaluate_grade(ref_metrics, preview_video)
            log_event(f"{shot_id} consistency: {after_score.overall_score}/100.")
            
            # Render Final Full-Quality Master
            final_lut = str(self.work_dir / f"{job_id}_{shot_id}_grade.cube")
            final_video = str(self.work_dir / f"{job_id}_{shot_id}_graded.mp4")
            log_event(f"Rendering final master video & 3D LUT for {shot_id}...")
            render_grade(path, params, final_video, final_lut, is_preview=False)
            
            if is_ref:
                explanation = (
                    f"Master technical reference. Applied creative '{creative_spec.look_title}' film stock emulation "
                    f"(Contrast: {creative_spec.contrast_intent}x, Saturation: {creative_spec.saturation_intent}x). "
                    f"{'Preserved natural skin tones. ' if skin_prot else ''}"
                )
            elif is_same_scene:
                explanation = (
                    f"Continuous scene match to {ref_shot_id} with unified '{creative_spec.look_title}' style. "
                    f"{'Protected skin tones. ' if skin_prot else ''}"
                    f"Consistency improved from {before_score.overall_score} to {after_score.overall_score}."
                )
            else:
                explanation = (
                    f"Scene-aware film emulation ({semantic.lighting_environment}). Preserved natural scene exposure "
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