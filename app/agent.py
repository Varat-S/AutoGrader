import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from app.models.analysis import (
    ShotSemanticAnalysis,
    ShotMetrics,
    CreativeSpecification,
    InputProfileAssessment,
    NormalizationValidationResult,
    LookContinuityScore,
    SceneHealthScore,
    SceneIntent
)
from app.models.grade import (
    ColorGradeParams,
    GradePlan,
    InputTransformParams,
    ConsistencyScore,
    GradeResult,
    RevisionRecord,
    SceneMatchParams,
    SceneTrimParams,
    ThreeWayTonalParams,
    TechnicalBalanceParams,
    EffectiveGradeSummary
)
from app.tools.inspect_footage import inspect_all_shots_batched
from app.tools.measure_color import measure_shot_color
from app.tools.research import research_cinematography_principles, synthesize_creative_specification
from app.tools.calculate_grade import build_grade_plan, assess_input_profile
from app.tools.render import render_grade
from app.tools.evaluate import evaluate_grade
from app.media.color import (
    is_log_profile,
    apply_color_grade_to_frame,
    compute_consistency_score,
    calculate_deterministic_match_params,
    aggregate_shot_metrics,
    assess_normalization_health,
    evaluate_transform_look_continuity,
    evaluate_scene_health
)
from app.media.lut import generate_3d_cube_lut, generate_shared_creative_look_lut
from app.media.ffmpeg import extract_sampled_frames, probe_video, generate_matched_browser_proxies

class SceneColoristAgent:
    """Dedicated autonomous colorist agent for an individual scene.
    
    Responsibilities:
    1. Independent Per-Scene Luminance: Decouples luminance/exposure across scenes,
       grading each scene's contrast and tone curve for its specific dramatic and physical context.
    2. Sequence Color Matching: Harmonizes chromaticity (a*, b* gains and offsets, white balance,
       and shared creative split-toning) against the sequence master reference.
    3. 3-Way Tonal Controls: Manipulates shadows (lift, tint, saturation), midtones (gamma,
       contrast, tint, saturation), and highlights (gain, roll-off shoulder, tint, saturation).
    4. Context-Aware Acceptance: Acknowledges and accepts blown-out skies in high-key/outdoor scenes
       and deep clipped shadows in low-key/interior scenes without penalty when aligned with artistic intent.
    5. Autonomous Review & Revision: Evaluates graded frames and performs targeted 3-way revisions.
    """
    def __init__(
        self,
        scene_id: str,
        creative_spec: CreativeSpecification,
        master_chroma_reference: ShotMetrics,
        master_ref_plan: GradePlan,
        log_callback: Optional[callable] = None
    ):
        self.scene_id = scene_id
        self.creative_spec = creative_spec
        self.master_chroma_reference = master_chroma_reference
        self.master_ref_plan = master_ref_plan
        self.log_callback = log_callback or (lambda msg: None)

    def grade_shot(
        self,
        shot_id: str,
        metrics: ShotMetrics,
        semantic: ShotSemanticAnalysis,
        cached_frames: List[np.ndarray],
        cached_timestamps: List[float],
        shot_profile: str,
        is_reference_shot: bool = False,
        balanced_ref_metrics: Optional[ShotMetrics] = None,
        norm_result: Optional[NormalizationValidationResult] = None,
        is_same_scene: bool = False
    ) -> Tuple[GradePlan, ConsistencyScore, ConsistencyScore, str, List[RevisionRecord], ShotMetrics]:
        shot_scene_intent = self.creative_spec.get_scene_intent(semantic.scene_group_id)

        if is_reference_shot:
            self.log_callback(f"  [{self.scene_id}] Finalizing and Validating Master Reference {shot_id}...")
            plan = self.master_ref_plan.model_copy(deep=True)
            if plan.three_way is None:
                plan.three_way = ThreeWayTonalParams()

            ref_preview_frames = [apply_color_grade_to_frame(f, plan) for f in cached_frames]
            eval_metrics, _ = evaluate_grade(
                reference_metrics=metrics,
                graded_video_or_frames=ref_preview_frames,
                evaluation_mode="same_scene_match",
                timestamps=cached_timestamps
            )

            ref_health = evaluate_scene_health(
                source_metrics=metrics,
                graded_metrics=eval_metrics,
                graded_frames=ref_preview_frames,
                scene_intent=shot_scene_intent
            )

            final_state = "ACCEPTED"
            if not ref_health.hard_gates_passed or not ref_health.passed:
                self.log_callback(f"  [Reference Health Notice] Master reference {shot_id} triggered health notice (Score: {ref_health.overall_score}/100, Gates: {ref_health.hard_gate_failures})")
                revised_ref_plan = plan.model_copy(deep=True)
                if revised_ref_plan.scene_trim is None:
                    revised_ref_plan.scene_trim = SceneTrimParams()
                if any("shadow" in f.lower() for f in getattr(ref_health, "hard_gate_failures", [])):
                    revised_ref_plan.scene_trim.trim_shadow_sat = 0.70
                if any("clip" in f.lower() for f in getattr(ref_health, "hard_gate_failures", [])):
                    revised_ref_plan.scene_trim.trim_contrast = max(0.80, round(revised_ref_plan.scene_trim.trim_contrast * 0.90, 2))
                
                rev_frames = [apply_color_grade_to_frame(f, revised_ref_plan) for f in cached_frames]
                rev_metrics, _ = evaluate_grade(
                    reference_metrics=metrics,
                    graded_video_or_frames=rev_frames,
                    evaluation_mode="same_scene_match",
                    timestamps=cached_timestamps
                )
                rev_health = evaluate_scene_health(metrics, rev_metrics, rev_frames, shot_scene_intent)
                if rev_health.hard_gates_passed and rev_health.passed:
                    plan = revised_ref_plan
                    eval_metrics = rev_metrics
                    ref_health = rev_health
                    self.log_callback(f"  [Reference Health Recovered] Master reference health satisfied after targeted trim.")
                else:
                    final_state = "WARNING_FLAGGED"
                    self.log_callback(f"  [Reference Health Warning] Master reference retained with {final_state}.")

            before_score = compute_consistency_score(
                reference=eval_metrics,
                candidate=metrics,
                evaluation_mode="reference_baseline",
                scene_intent=shot_scene_intent,
                source_metrics=metrics
            )
            after_score = compute_consistency_score(
                reference=eval_metrics,
                candidate=eval_metrics,
                evaluation_mode="reference_baseline",
                scene_intent=shot_scene_intent,
                graded_frames=ref_preview_frames,
                source_metrics=metrics
            )
            history = [RevisionRecord(
                iteration=0,
                state=final_state,
                action_taken="Master technical reference standard established." if final_state == "ACCEPTED" else f"Master reference standard established with {final_state}.",
                overall_score_before=after_score.overall_score,
                overall_score_after=after_score.overall_score,
                diagnosis=after_score.diagnosis or "Reference baseline",
                parameter_deltas={}
            )]
            return (plan, before_score, after_score, final_state, history, eval_metrics)

        # Candidate Shot Grading
        self.log_callback(f"  [{self.scene_id}] Grading shot {shot_id} (Context: {semantic.lighting_environment}, Lighting: {semantic.time_of_day})...")
        
        # 1. Context Analysis: Detect skies/daylight and shadows/low-key
        env_lower = semantic.lighting_environment.lower()
        desc_lower = semantic.scene_description.lower()
        intent_light = getattr(shot_scene_intent, "lighting_class", "").lower() if shot_scene_intent else ""
        
        is_sky_or_high_key = any(k in env_lower or k in desc_lower or k in intent_light for k in ["sky", "daylight", "sun", "aerial", "outdoor", "high_key", "bright"])
        is_shadow_or_low_key = any(k in env_lower or k in desc_lower or k in intent_light for k in ["night", "low_key", "dark", "interior", "silhouette", "moody", "shadow"])
        
        if is_sky_or_high_key:
            self.log_callback(f"  [{self.scene_id}] High-key / sky context detected: highlight roll-off and sky blowout accepted as intentional artistic choice.")
        if is_shadow_or_low_key:
            self.log_callback(f"  [{self.scene_id}] Low-key / shadow context detected: deep shadow crush accepted as intentional artistic depth.")

        # 2. Build initial candidate plan
        initial_cand_plan = build_grade_plan(
            reference=self.master_chroma_reference,
            target=metrics,
            target_semantic=semantic,
            creative_spec=self.creative_spec,
            is_reference_shot=False,
            is_same_scene=is_same_scene,
            color_profile=shot_profile,
            matched_params=None,
            scene_intent=shot_scene_intent
        )

        # 3. Shot Match: Bounded Luminance Matching for Same-Scene, Natural Exposure for Independent Scenes
        cand_balanced_frames = [apply_color_grade_to_frame(f, GradePlan(
            shot_id=f"{shot_id}_balanced",
            input_transform=initial_cand_plan.input_transform,
            technical_balance=initial_cand_plan.technical_balance
        )) for f in cached_frames]
        
        balanced_cand_metrics = aggregate_shot_metrics(
            shot_id=f"{shot_id}_balanced",
            video_path="",
            frames=cand_balanced_frames,
            timestamps=cached_timestamps,
            fps=metrics.fps,
            width=metrics.width,
            height=metrics.height,
            duration_sec=metrics.duration_sec
        )
        
        if is_same_scene:
            # SAME-SCENE: Match luminance with bounded protection against clipping & composition disparity
            bal_ref = balanced_ref_metrics if balanced_ref_metrics is not None else self.master_chroma_reference
            residual_match = calculate_deterministic_match_params(
                reference=bal_ref,
                target=balanced_cand_metrics,
                strength=0.95,
                match_luminance=True
            )
            initial_cand_plan.scene_match = SceneMatchParams(
                lab_l_gain=float(np.clip(residual_match.lab_l_gain, 0.5, 2.0)),
                lab_l_offset=float(np.clip(residual_match.lab_l_offset, -60.0, 60.0)),
                lab_a_gain=float(np.clip(residual_match.lab_a_gain, 0.4, 2.5)),
                lab_a_offset=float(np.clip(residual_match.lab_a_offset, -80.0, 80.0)),
                lab_b_gain=float(np.clip(residual_match.lab_b_gain, 0.4, 2.5)),
                lab_b_offset=float(np.clip(residual_match.lab_b_offset, -80.0, 80.0))
            )
            self.log_callback(f"  [{self.scene_id}] Same-scene shot: restored bounded luminance matching (L_gain={initial_cand_plan.scene_match.lab_l_gain:.2f}, L_offset={initial_cand_plan.scene_match.lab_l_offset:+.1f})")
        else:
            # INDEPENDENT SCENE: Preserve natural exposure & lighting, match only chromaticity & creative look
            residual_match = calculate_deterministic_match_params(
                reference=self.master_chroma_reference,
                target=balanced_cand_metrics,
                strength=0.95,
                match_luminance=False
            )
            initial_cand_plan.scene_match = SceneMatchParams(
                lab_l_gain=1.0,
                lab_l_offset=0.0,
                lab_a_gain=float(np.clip(residual_match.lab_a_gain, 0.4, 2.5)),
                lab_a_offset=float(np.clip(residual_match.lab_a_offset, -80.0, 80.0)),
                lab_b_gain=float(np.clip(residual_match.lab_b_gain, 0.4, 2.5)),
                lab_b_offset=float(np.clip(residual_match.lab_b_offset, -80.0, 80.0))
            )
            self.log_callback(f"  [{self.scene_id}] Independent scene: natural exposure preserved (L_gain=1.0, L_offset=0.0), matching colors only")

        # 4. Context-Aware 3-Way Tonal Manipulation: Kept neutral unless genuinely additional zonal correction needed
        if initial_cand_plan.three_way is None:
            initial_cand_plan.three_way = ThreeWayTonalParams()
            
        if is_sky_or_high_key:
            initial_cand_plan.three_way.highlight_rolloff = 0.88 # Soft filmic roll-off for sky
        if is_shadow_or_low_key:
            initial_cand_plan.three_way.midtone_gamma = 1.05 # Modest midtone gamma lift for shadow readability

        eval_mode = "same_scene_match" if is_same_scene else "cross_scene_look_continuity"
        match_colors_only = not is_same_scene

        before_score = compute_consistency_score(
            reference=self.master_chroma_reference,
            candidate=metrics,
            evaluation_mode=eval_mode,
            ref_plan=self.master_ref_plan,
            cand_plan=None,
            scene_intent=shot_scene_intent,
            source_metrics=metrics,
            match_colors_only=match_colors_only
        )

        # 5. Review & Revision State Machine
        initial_preview_frames = [apply_color_grade_to_frame(f, initial_cand_plan) for f in cached_frames]
        eval_metrics, initial_score = evaluate_grade(
            reference_metrics=self.master_chroma_reference,
            graded_video_or_frames=initial_preview_frames,
            evaluation_mode=eval_mode,
            timestamps=cached_timestamps,
            ref_plan=self.master_ref_plan,
            cand_plan=initial_cand_plan,
            scene_intent=shot_scene_intent,
            source_metrics=metrics,
            match_colors_only=match_colors_only
        )
        init_look_score = evaluate_transform_look_continuity(self.master_ref_plan, initial_cand_plan, eval_metrics)
        init_health_score = evaluate_scene_health(metrics, eval_metrics, initial_preview_frames, shot_scene_intent)

        best_plan = initial_cand_plan.model_copy(deep=True)
        best_score = initial_score
        best_look = init_look_score
        best_health = init_health_score
        history: List[RevisionRecord] = []
        revisions_performed = 0
        max_revisions = 2
        rejected_deltas = []

        self.log_callback(f"  [{self.scene_id}] Initial grade: overall {initial_score.overall_score}/100 (Tone: {initial_score.tonal_similarity}, Chroma: {initial_score.chromatic_similarity}, Health: {init_health_score.overall_score}/100, SkyAccepted={init_health_score.artistic_sky_clipping_accepted}, ShadowAccepted={init_health_score.artistic_shadow_clipping_accepted})")

        is_initially_accepted = (
            (is_same_scene and initial_score.overall_score >= 75.0 and init_health_score.passed and init_health_score.hard_gates_passed) or
            (not is_same_scene and initial_score.overall_score >= 75.0 and init_look_score.overall_score >= 75.0 and init_health_score.passed and init_health_score.hard_gates_passed)
        )
        is_accepted = is_initially_accepted

        if is_initially_accepted:
            final_state = "ACCEPTED"
            history.append(RevisionRecord(
                iteration=0,
                state="ACCEPTED",
                action_taken="Initial grade plan satisfies color matching and scene health tolerance.",
                overall_score_before=initial_score.overall_score,
                overall_score_after=initial_score.overall_score,
                diagnosis=initial_score.diagnosis or "Pass",
                parameter_deltas={}
            ))
            self.log_callback(f"  [State] {shot_id} -> ACCEPTED on initial evaluation.")
        else:
            final_state = "INITIAL_EVALUATION"
            history.append(RevisionRecord(
                iteration=0,
                state="INITIAL_EVALUATION",
                action_taken=f"Initial score below threshold ({initial_score.overall_score} < 75.0 or health gate active). Beginning diagnostic revisions.",
                overall_score_before=initial_score.overall_score,
                overall_score_after=initial_score.overall_score,
                diagnosis=initial_score.diagnosis or "Discrepancy detected",
                parameter_deltas={}
            ))

            for rev_idx in range(1, max_revisions + 1):
                proposed_plan = best_plan.model_copy(deep=True)
                if proposed_plan.scene_trim is None:
                    proposed_plan.scene_trim = SceneTrimParams()
                if proposed_plan.three_way is None:
                    proposed_plan.three_way = ThreeWayTonalParams()
                deltas = {}
                action_desc = ""

                hard_failures = getattr(best_health, "hard_gate_failures", [])
                has_shadow_sat_fail = (best_health.shadow_saturation_health < 80.0 or any("shadow" in f.lower() for f in hard_failures))
                has_clip_fail = (not is_sky_or_high_key and not is_shadow_or_low_key and (best_score.clipping_health < 80.0 or best_health.clipping_health < 80.0 or any("clip" in f.lower() for f in hard_failures)))

                if is_same_scene:
                    # Same-Scene Diagnostic Policy: mutate technical_balance or scene_trim
                    if best_score.tonal_similarity < 70.0:
                        p50_target = self.master_chroma_reference.p50_luminance if self.master_chroma_reference.p50_luminance > 0 else self.master_chroma_reference.avg_luminance
                        p50_cand = eval_metrics.p50_luminance if eval_metrics.p50_luminance > 0 else eval_metrics.avg_luminance
                        ev_adj = float(np.clip(np.log2(max(1.0, p50_target) / max(1.0, p50_cand)) * 0.45, -1.0, 1.0))
                        if abs(ev_adj) > 0.02:
                            new_ev = float(np.clip(proposed_plan.technical_balance.exposure_ev + ev_adj, -2.5, 2.5))
                            d_ev = round(new_ev - proposed_plan.technical_balance.exposure_ev, 2)
                            if ("exposure_ev", d_ev) not in rejected_deltas and abs(d_ev) > 0.02:
                                deltas["exposure_ev"] = d_ev
                                proposed_plan.technical_balance.exposure_ev = round(new_ev, 2)
                                action_desc = f"Adjusted technical exposure by {d_ev:+.2f} EV"

                        if not deltas:
                            ref_iqr = self.master_chroma_reference.p75_luminance - self.master_chroma_reference.p25_luminance
                            cand_iqr = eval_metrics.p75_luminance - eval_metrics.p25_luminance
                            if ref_iqr > 10.0 and cand_iqr > 0.0:
                                iqr_ratio = ref_iqr / max(5.0, cand_iqr)
                                if iqr_ratio > 1.2:
                                    new_c = min(1.30, round(proposed_plan.scene_trim.trim_contrast * 1.10, 2))
                                    d_c = round(new_c - proposed_plan.scene_trim.trim_contrast, 2)
                                    if abs(d_c) > 0.02 and ("trim_contrast", d_c) not in rejected_deltas:
                                        deltas["trim_contrast"] = d_c
                                        proposed_plan.scene_trim.trim_contrast = new_c
                                        action_desc = f"Steepened scene trim contrast to {new_c}x to match tonal spread"
                                elif iqr_ratio < 0.8:
                                    new_c = max(0.75, round(proposed_plan.scene_trim.trim_contrast * 0.90, 2))
                                    d_c = round(new_c - proposed_plan.scene_trim.trim_contrast, 2)
                                    if abs(d_c) > 0.02 and ("trim_contrast", d_c) not in rejected_deltas:
                                        deltas["trim_contrast"] = d_c
                                        proposed_plan.scene_trim.trim_contrast = new_c
                                        action_desc = f"Softened scene trim contrast to {new_c}x to match tonal spread"

                    if not deltas and best_score.chromatic_similarity < 70.0:
                        delta_b = self.master_chroma_reference.avg_lab_mean[2] - eval_metrics.avg_lab_mean[2]
                        delta_a = self.master_chroma_reference.avg_lab_mean[1] - eval_metrics.avg_lab_mean[1]
                        t_adj = float(np.clip(delta_b * 0.35, -15.0, 15.0))
                        tint_adj = float(np.clip(delta_a * 0.35, -10.0, 10.0))
                        if abs(t_adj) > 0.5 or abs(tint_adj) > 0.5:
                            new_temp = float(np.clip(proposed_plan.technical_balance.temperature + t_adj, -40.0, 40.0))
                            new_tint = float(np.clip(proposed_plan.technical_balance.tint + tint_adj, -25.0, 25.0))
                            d_t = round(new_temp - proposed_plan.technical_balance.temperature, 1)
                            d_tint = round(new_tint - proposed_plan.technical_balance.tint, 1)
                            if ("temperature", d_t) not in rejected_deltas:
                                deltas["temperature"] = d_t
                                deltas["tint"] = d_tint
                                proposed_plan.technical_balance.temperature = round(new_temp, 1)
                                proposed_plan.technical_balance.tint = round(new_tint, 1)
                                action_desc = f"Refined white balance (temp: {d_t:+.1f}, tint: {d_tint:+.1f})"

                    if not deltas and not is_sky_or_high_key and not is_shadow_or_low_key and (best_score.clipping_health < 80.0 or any("clip" in f.lower() for f in hard_failures)):
                        new_c = max(0.75, round(proposed_plan.scene_trim.trim_contrast * 0.90, 2))
                        d_c = round(new_c - proposed_plan.scene_trim.trim_contrast, 2)
                        if abs(d_c) > 0.02 and ("trim_contrast", d_c) not in rejected_deltas:
                            deltas["trim_contrast"] = d_c
                            proposed_plan.scene_trim.trim_contrast = new_c
                            action_desc = f"Softened scene trim contrast to {new_c}x to eliminate clipping"
                        elif proposed_plan.scene_trim.trim_shadow_lift < 2.0:
                            new_lift = min(3.0, round(proposed_plan.scene_trim.trim_shadow_lift + 0.5, 2))
                            deltas["trim_shadow_lift"] = 0.5
                            proposed_plan.scene_trim.trim_shadow_lift = new_lift
                            action_desc = f"Lifted shadow toe by {new_lift} to recover clipped blacks"
                else:
                    # Cross-Scene Diagnostic Policy: mutate ONLY scene_trim or technical_balance
                    if has_shadow_sat_fail:
                        curr_sh_sat = getattr(proposed_plan.scene_trim, "trim_shadow_sat", 1.0)
                        new_sh_sat = max(0.40, round(curr_sh_sat * 0.80, 2))
                        d_sh_sat = round(new_sh_sat - curr_sh_sat, 2)
                        if abs(d_sh_sat) > 0.02 and ("trim_shadow_sat", d_sh_sat) not in rejected_deltas:
                            deltas["trim_shadow_sat"] = d_sh_sat
                            proposed_plan.scene_trim.trim_shadow_sat = new_sh_sat
                            action_desc = f"Trimmed shadow saturation to {new_sh_sat}x to protect shadow health"
                        else:
                            new_sat = max(0.65, round(proposed_plan.scene_trim.trim_saturation * 0.85, 2))
                            d_sat = round(new_sat - proposed_plan.scene_trim.trim_saturation, 2)
                            if abs(d_sat) > 0.02 and ("trim_saturation", d_sat) not in rejected_deltas:
                                deltas["trim_saturation"] = d_sat
                                proposed_plan.scene_trim.trim_saturation = new_sat
                                action_desc = f"Reduced scene trim saturation to {new_sat}x to protect shadow health"
                    elif has_clip_fail:
                        new_c = max(0.75, round(proposed_plan.scene_trim.trim_contrast * 0.90, 2))
                        d_c = round(new_c - proposed_plan.scene_trim.trim_contrast, 2)
                        if abs(d_c) > 0.02 and ("trim_contrast", d_c) not in rejected_deltas:
                            deltas["trim_contrast"] = d_c
                            proposed_plan.scene_trim.trim_contrast = new_c
                            action_desc = f"Softened scene trim contrast to {new_c}x to eliminate clipping"
                        elif proposed_plan.scene_trim.trim_shadow_lift < 2.0:
                            new_lift = min(3.0, round(proposed_plan.scene_trim.trim_shadow_lift + 0.5, 2))
                            deltas["trim_shadow_lift"] = 0.5
                            proposed_plan.scene_trim.trim_shadow_lift = new_lift
                            action_desc = f"Lifted shadow toe by {new_lift} to recover clipped shadows"
                    elif best_health.exposure_appropriateness < 80.0 and shot_scene_intent:
                        min_ev, max_ev = shot_scene_intent.source_relative_exposure_bounds
                        curr_ev = best_health.source_relative_exposure_change_ev
                        if curr_ev < min_ev:
                            corr = min_ev - curr_ev
                            new_trim_ev = round(proposed_plan.scene_trim.trim_exposure_ev + corr, 2)
                            deltas["trim_exposure_ev"] = round(new_trim_ev - proposed_plan.scene_trim.trim_exposure_ev, 2)
                            proposed_plan.scene_trim.trim_exposure_ev = new_trim_ev
                            action_desc = f"Lifted scene trim exposure by {corr:+.2f} EV into scene bounds"
                        elif curr_ev > max_ev:
                            corr = max_ev - curr_ev
                            new_trim_ev = round(proposed_plan.scene_trim.trim_exposure_ev + corr, 2)
                            deltas["trim_exposure_ev"] = round(new_trim_ev - proposed_plan.scene_trim.trim_exposure_ev, 2)
                            proposed_plan.scene_trim.trim_exposure_ev = new_trim_ev
                            action_desc = f"Lowered scene trim exposure by {corr:+.2f} EV into scene bounds"

                if not deltas:
                    history.append(RevisionRecord(
                        iteration=revisions_performed,
                        state="NO_ACTIONABLE_REVISION",
                        action_taken="No further actionable parameter adjustment diagnosed.",
                        overall_score_before=best_score.overall_score,
                        overall_score_after=best_score.overall_score,
                        diagnosis=best_score.diagnosis or "Unchanged",
                        parameter_deltas={}
                    ))
                    final_state = "NO_ACTIONABLE_REVISION"
                    self.log_callback(f"  [State] {shot_id} -> NO_ACTIONABLE_REVISION. Retaining best plan.")
                    break

                revisions_performed += 1
                self.log_callback(f"  [Revision {revisions_performed} PROPOSED] {action_desc}...")
                prop_preview_frames = [apply_color_grade_to_frame(f, proposed_plan) for f in cached_frames]
                prop_metrics, prop_score = evaluate_grade(
                    reference_metrics=self.master_chroma_reference,
                    graded_video_or_frames=prop_preview_frames,
                    evaluation_mode=eval_mode,
                    timestamps=cached_timestamps,
                    ref_plan=self.master_ref_plan,
                    cand_plan=proposed_plan,
                    scene_intent=shot_scene_intent,
                    source_metrics=metrics,
                    match_colors_only=match_colors_only
                )
                prop_look = evaluate_transform_look_continuity(self.master_ref_plan, proposed_plan, prop_metrics)
                prop_health = evaluate_scene_health(metrics, prop_metrics, prop_preview_frames, shot_scene_intent)

                is_better = False
                if best_health.hard_gates_passed and not prop_health.hard_gates_passed:
                    is_better = False
                else:
                    continuity_drop = best_score.overall_score - prop_score.overall_score
                    best_gates = len(getattr(best_health, "hard_gate_failures", [])) if not best_health.hard_gates_passed else 0
                    prop_gates = len(getattr(prop_health, "hard_gate_failures", [])) if not prop_health.hard_gates_passed else 0

                    if prop_gates < best_gates:
                        is_better = (continuity_drop <= 10.0)
                    elif prop_gates > best_gates:
                        is_better = False
                    elif prop_health.passed and not best_health.passed:
                        is_better = (continuity_drop <= 5.0)
                    elif best_health.passed and not prop_health.passed:
                        is_better = False
                    elif prop_health.passed == best_health.passed:
                        health_diff = prop_health.overall_score - best_health.overall_score
                        if health_diff > 1.0:
                            is_better = (continuity_drop <= 4.0)
                        elif abs(health_diff) <= 1.0:
                            score_diff = prop_score.overall_score - best_score.overall_score
                            if score_diff > 0.5:
                                is_better = True
                            elif abs(score_diff) <= 0.2:
                                def _plan_delta_mag(p: GradePlan) -> float:
                                    m = abs(p.technical_balance.exposure_ev) + abs(p.technical_balance.temperature / 10.0) + abs(p.technical_balance.tint / 10.0)
                                    if p.scene_trim:
                                        m += abs(p.scene_trim.trim_exposure_ev) + abs(p.scene_trim.trim_contrast - 1.0) + abs(p.scene_trim.trim_saturation - 1.0)
                                    return m
                                if _plan_delta_mag(proposed_plan) < _plan_delta_mag(best_plan) - 1e-4:
                                    is_better = True

                if is_better:
                    history.append(RevisionRecord(
                        iteration=revisions_performed,
                        state="REVISION_IMPROVED",
                        action_taken=f"{action_desc} (improved score: {best_score.overall_score} -> {prop_score.overall_score})",
                        overall_score_before=best_score.overall_score,
                        overall_score_after=prop_score.overall_score,
                        diagnosis=prop_score.diagnosis or "Improved",
                        parameter_deltas=deltas
                    ))
                    best_plan = proposed_plan.model_copy(deep=True)
                    best_score = prop_score
                    best_look = prop_look
                    best_health = prop_health
                    eval_metrics = prop_metrics
                    self.log_callback(f"  [State] Revision {revisions_performed} IMPROVED score to {best_score.overall_score}/100. Updated best plan.")

                    is_accepted = (
                        (is_same_scene and best_score.overall_score >= 75.0 and best_health.passed and best_health.hard_gates_passed) or
                        (not is_same_scene and best_score.overall_score >= 75.0 and best_look.overall_score >= 75.0 and best_health.passed and best_health.hard_gates_passed)
                    )
                    if is_accepted:
                        final_state = "ACCEPTED"
                        self.log_callback(f"  [State] {shot_id} -> ACCEPTED.")
                        break
                else:
                    for k, v in deltas.items():
                        rejected_deltas.append((k, v))
                    history.append(RevisionRecord(
                        iteration=revisions_performed,
                        state="REVISION_REJECTED",
                        action_taken=f"{action_desc} rejected (score did not improve: {prop_score.overall_score} vs best {best_score.overall_score})",
                        overall_score_before=best_score.overall_score,
                        overall_score_after=prop_score.overall_score,
                        diagnosis=prop_score.diagnosis or "Reverted",
                        parameter_deltas=deltas
                    ))
                    self.log_callback(f"  [State] Revision {revisions_performed} REJECTED ({prop_score.overall_score} vs best {best_score.overall_score}). Reverted to best plan.")

            if final_state == "NO_ACTIONABLE_REVISION":
                pass
            elif not is_accepted and (not best_health.passed or not best_health.hard_gates_passed):
                final_state = "BEST_EFFORT_HEALTH_WARNING"
                self.log_callback(f"  [Warning: Health Gate Active] {shot_id} -> BEST_EFFORT_HEALTH_WARNING (final score: {best_score.overall_score}/100).")
            elif final_state not in ["ACCEPTED", "NO_ACTIONABLE_REVISION"]:
                final_state = "MAX_REVISIONS_REACHED"
                self.log_callback(f"  [State] {shot_id} -> MAX_REVISIONS_REACHED (final score: {best_score.overall_score}/100).")

        return (best_plan, before_score, best_score, final_state, history, eval_metrics)

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
        input_profiles: Optional[List[Any]] = None,
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
            
        # Determine Master Reference Shot (Gemini recommended or user override)
        if reference_index is not None and 0 <= reference_index < len(video_paths):
            ref_idx = reference_index
            log_event(f"Using user-specified master reference shot: shot_{chr(65 + ref_idx)}")
        else:
            rec_id = inspection_result.recommended_reference_shot_id
            ref_idx = 0
            for idx, s in enumerate(semantic_analyses):
                if s.shot_id == rec_id:
                    ref_idx = idx
                    break
                    
        ref_shot_id = f"shot_{chr(65 + ref_idx)}"
        ref_semantic = semantic_analyses[ref_idx]
        ref_group_id = ref_semantic.scene_group_id
        
        # P1: Dynamically recompute scene relationships relative to final reference shot
        for idx, sem in enumerate(semantic_analyses):
            if idx == ref_idx:
                sem.relationship_to_reference = "reference"
            elif sem.scene_group_id == ref_group_id:
                sem.relationship_to_reference = "same_scene"
            else:
                sem.relationship_to_reference = "independent_scene"
                
        log_event(f"Selected '{ref_shot_id}' as master technical reference (Group: {ref_semantic.scene_group_id}, Context: {ref_semantic.lighting_environment}).")
        
        # 2. MEASURE: Extract numerical color statistics & Sample Keyframes & Assess Profiles
        log_event("Measuring numerical color distributions and assessing camera profiles...")
        
        # Resolve per-shot profile selections
        per_shot_profiles = {}
        per_shot_confirmed = {}
        per_shot_overrides = {}
        if input_profiles:
            for p_sel in input_profiles:
                idx = p_sel.get("shot_index", 0) if isinstance(p_sel, dict) else getattr(p_sel, "shot_index", 0)
                prof = p_sel.get("profile", "rec709") if isinstance(p_sel, dict) else getattr(p_sel, "profile", "rec709")
                conf = p_sel.get("user_confirmed", False) if isinstance(p_sel, dict) else getattr(p_sel, "user_confirmed", False)
                ovr = p_sel.get("override_warning", False) if isinstance(p_sel, dict) else getattr(p_sel, "override_warning", False)
                prof_str = prof.value if hasattr(prof, "value") else str(prof)
                per_shot_profiles[idx] = prof_str
                per_shot_confirmed[idx] = conf
                per_shot_overrides[idx] = ovr

        shot_metrics: List[ShotMetrics] = []
        shot_assessments: List[InputProfileAssessment] = []
        cached_frames: List[List[np.ndarray]] = []
        cached_timestamps: List[List[float]] = []
        resolved_profiles: List[str] = []
        probed_infos: List[Dict[str, Any]] = []
        
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            probed_info = probe_video(path)
            probed_infos.append(probed_info)
            metrics = measure_shot_color(path, shot_id=shot_id)
            shot_metrics.append(metrics)
            
            frames, timestamps = extract_sampled_frames(path, num_samples=6)
            cached_frames.append(frames)
            cached_timestamps.append(timestamps)
            
            req_prof = per_shot_profiles.get(i, color_profile)
            is_conf = per_shot_confirmed.get(i, False)
            assessment = assess_input_profile(shot_id, probed_info, metrics, requested_profile=req_prof, shot_index=i, user_confirmed=is_conf)
            shot_assessments.append(assessment)
            
            final_prof = assessment.resolved_profile or assessment.selected_profile
            if final_prof == "auto_ask":
                raise ValueError(f"Unresolved profile 'auto_ask' for shot {shot_id}. Every shot must have a concrete resolved profile before grading begins.")
            resolved_profiles.append(final_prof)
            
            rec_str = assessment.recommended_profile or "None"
            log_event(f"  [Input Profile] {shot_id}: Requested='{req_prof}' | Recommended='{rec_str}' | Resolved='{final_prof}' | Source='{assessment.resolution_source}'")
            if assessment.profile_mismatch_warning:
                log_event(f"    [Safety Warning] {shot_id}: {assessment.warning_message}")
            
        ref_metrics = shot_metrics[ref_idx]
        ref_profile = resolved_profiles[ref_idx]
        
        # 2b. PREFLIGHT NORMALIZATION GATE (Execute on ALL sequence shots BEFORE paid LLM calls)
        log_event("Executing preflight normalization safety gates across all sequence shots...")
        normalization_results: List[NormalizationValidationResult] = []
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            prof = resolved_profiles[i]
            metrics = shot_metrics[i]
            probed_info = probed_infos[i]
            
            norm_plan = GradePlan(
                shot_id=f"{shot_id}_preflight_norm",
                input_transform=InputTransformParams(
                    is_log=is_log_profile(prof),
                    profile=prof
                )
            )
            norm_frames = [apply_color_grade_to_frame(f, norm_plan) for f in cached_frames[i]]
            norm_res = assess_normalization_health(
                shot_id=shot_id,
                source_metrics=metrics,
                normalized_frames=norm_frames,
                profile=prof,
                probed_info=probed_info
            )
            
            shot_has_override = per_shot_overrides.get(i, False) or per_shot_confirmed.get(i, False)
            norm_res.passed = True  # Non-blocking: never halt delivery
            if shot_has_override and norm_res.state != "NORMALIZATION_VERIFIED":
                norm_res.state = "NORMALIZATION_WARNING_OVERRIDDEN"
            
            normalization_results.append(norm_res)
            log_event(f"  [Preflight Normalization] {shot_id}: {norm_res.state} — {norm_res.reason}")

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
            research_result=research_result,
            scene_analyses=semantic_analyses
        )
        creative_spec.normalize_canonical_look()
        if creative_spec.synthesis_mode == "fallback":
            reason = getattr(creative_spec, "fallback_reason", None)
            if reason:
                log_event(f"  [API Quota/Availability Notice] Creative synthesis fell back to neutral standard: {reason[:140]}")
            else:
                log_event("Creative synthesis fell back to neutral photographic standard (no artificial color bias).")
        else:
            log_event(f"Synthesized Look ({creative_spec.synthesis_mode}): '{creative_spec.look_title}' (Contrast: {creative_spec.contrast_intent}x, Saturation: {creative_spec.saturation_intent}x, Highlights: {creative_spec.highlight_bias}, Shadows: {creative_spec.shadow_bias})")
        
        # Export Shared Creative-Look LUT
        shared_lut_path = str(self.work_dir / f"{job_id}_shared_creative_look.cube")
        generate_shared_creative_look_lut(creative_spec, shared_lut_path)
        log_event(f"Exported Shared Creative-Look 3D LUT: {Path(shared_lut_path).name}")
        
        # 4. GRADE MASTER REFERENCE FIRST & ESTABLISH REFERENCE TARGET METRICS
        log_event(f"Grading Master Reference {ref_shot_id} with creative look '{creative_spec.look_title}'...")
        ref_scene_intent = creative_spec.get_scene_intent(ref_semantic.scene_group_id)
        ref_plan = build_grade_plan(
            reference=ref_metrics,
            target=ref_metrics,
            target_semantic=ref_semantic,
            creative_spec=creative_spec,
            is_reference_shot=True,
            is_same_scene=False,
            color_profile=ref_profile,
            scene_intent=ref_scene_intent
        )
        
        # Master Reference Preflight Normalization status
        ref_norm_res = normalization_results[ref_idx]
        log_event(f"  [Normalization Gate] {ref_shot_id}: {ref_norm_res.state} — {ref_norm_res.reason}")
        
        # Grade reference sampled frames to establish true target metrics
        graded_ref_frames = [apply_color_grade_to_frame(f, ref_plan) for f in cached_frames[ref_idx]]
        graded_ref_metrics, _ = evaluate_grade(
            reference_metrics=ref_metrics,
            graded_video_or_frames=graded_ref_frames,
            evaluation_mode="same_scene_match",
            timestamps=cached_timestamps[ref_idx]
        )
        log_event(f"Established Graded Reference Target Metrics: Luminance={graded_ref_metrics.avg_luminance}, CIELAB={graded_ref_metrics.avg_lab_mean}")
        
        # Measure intermediate balanced reference frames for same-scene matching in active space
        ref_balanced_frames = [apply_color_grade_to_frame(f, GradePlan(
            shot_id=f"{ref_shot_id}_bal",
            input_transform=ref_plan.input_transform,
            technical_balance=ref_plan.technical_balance
        )) for f in cached_frames[ref_idx]]
        
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
        
        # 5. GRADE ALL SHOTS IN SEQUENCE WITH DEDICATED SCENE COLORIST AGENTS
        graded_results: List[GradeResult] = []
        scene_agents: Dict[str, SceneColoristAgent] = {}
        
        for i, path in enumerate(video_paths):
            shot_id = f"shot_{chr(65 + i)}"
            metrics = shot_metrics[i]
            semantic = semantic_analyses[i]
            shot_profile = resolved_profiles[i]
            is_ref = (i == ref_idx)
            
            # Explicit Per-Shot Scene Grouping
            is_same_scene = not is_ref and (
                semantic.relationship_to_reference == "same_scene" or
                semantic.scene_group_id == ref_semantic.scene_group_id
            )
            eval_mode = "reference_baseline" if is_ref else ("same_scene_match" if is_same_scene else "cross_scene_look_continuity")
            shot_scene_intent = creative_spec.get_scene_intent(semantic.scene_group_id)
            
            # Deploy dedicated SceneColoristAgent for each scene (1 agent per scene)
            scene_key = semantic.scene_group_id or f"scene_{shot_id}"
            if scene_key not in scene_agents:
                log_event(f"Deploying dedicated SceneColoristAgent for Scene '{scene_key}'...")
                scene_agents[scene_key] = SceneColoristAgent(
                    scene_id=scene_key,
                    creative_spec=creative_spec,
                    master_chroma_reference=graded_ref_metrics,
                    master_ref_plan=ref_plan,
                    log_callback=log_event
                )
                
            agent = scene_agents[scene_key]
            
            # Dedicated Scene Agent executes independent luminance grading & sequence color matching
            best_plan, before_score, after_score, final_state, history, eval_metrics = agent.grade_shot(
                shot_id=shot_id,
                metrics=metrics,
                semantic=semantic,
                cached_frames=cached_frames[i],
                cached_timestamps=cached_timestamps[i],
                shot_profile=shot_profile,
                is_reference_shot=is_ref,
                balanced_ref_metrics=balanced_ref_metrics if is_same_scene else None,
                norm_result=normalization_results[i],
                is_same_scene=is_same_scene
            )
            revisions_performed = len([h for h in history if h.iteration > 0])
            
            # Render Final Delivery Video & 3D LUT using best_plan
            final_lut = str(self.work_dir / f"{job_id}_{shot_id}_grade.cube")
            final_video = str(self.work_dir / f"{job_id}_{shot_id}_graded.mp4")
            log_event(f"Rendering final master delivery video & 3D LUT for {shot_id} (State: {final_state})...")
            render_grade(path, best_plan, final_video, final_lut, is_preview=False, is_log=False)

            # Render Matched Browser Proxies (Before without LUT vs After with LUT)
            before_proxy = str(self.work_dir / f"{job_id}_{shot_id}_before_proxy.mp4")
            after_proxy = str(self.work_dir / f"{job_id}_{shot_id}_after_proxy.mp4")
            log_event(f"Generating matched browser proxies for {shot_id}...")
            generate_matched_browser_proxies(
                source_path=path,
                lut_path=final_lut,
                before_proxy_path=before_proxy,
                after_proxy_path=after_proxy,
                max_height=720
            )

            # Compute effective grade summary and final scores
            active_intent = ref_scene_intent if is_ref else shot_scene_intent
            scene_rat = (getattr(active_intent, "concise_rationale", None) or getattr(active_intent, "rationale", "")) if active_intent else ""
            eff_summary = best_plan.compute_effective_summary(
                scene_group_id=semantic.scene_group_id,
                scene_class=active_intent.lighting_class if active_intent else "daylight",
                scene_rationale=scene_rat,
                camera_profile=shot_profile,
                revision_state=final_state,
                highlight_bias=creative_spec.highlight_bias,
                shadow_bias=creative_spec.shadow_bias
            )
            final_preview_frames = [apply_color_grade_to_frame(f, best_plan) for f in cached_frames[i]]
            final_look = evaluate_transform_look_continuity(ref_plan, best_plan, eval_metrics)
            final_health = evaluate_scene_health(
                source_metrics=metrics,
                graded_metrics=eval_metrics,
                graded_frames=final_preview_frames,
                scene_intent=shot_scene_intent if not is_ref else ref_scene_intent
            )
            
            # Construct honest explanation
            prof_desc = f"Input Profile: {shot_profile}. "
            if is_ref:
                explanation = f"Master technical reference shot established standard. {prof_desc}Creative Look '{creative_spec.look_title}' applied ({creative_spec.contrast_intent}x contrast, {creative_spec.highlight_bias} highlights, {creative_spec.shadow_bias} shadows)."
            elif is_same_scene:
                explanation = f"Matched to continuous scene reference {ref_shot_id} (Group {semantic.scene_group_id}). {prof_desc}Harmonized chromatic balance (score: {after_score.chromatic_similarity}) and tonal exposure (score: {after_score.tonal_similarity}). State: {final_state} over {revisions_performed} revisions."
            else:
                explanation = f"Independent scene group {semantic.scene_group_id} ({semantic.lighting_environment}, {semantic.time_of_day}). {prof_desc}Preserved natural scene exposure and shadow depth while aligning creative look transform invariants (continuity score: {after_score.overall_score}). State: {final_state}."
                
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
                explanation=explanation,
                evaluation_mode=eval_mode,
                grade_summary=eff_summary,
                look_continuity=final_look,
                scene_health=final_health,
                before_proxy_path=before_proxy,
                after_proxy_path=after_proxy,
                original_source_path=path
            ))
            
        log_event(f"Autonomous color grading completed for sequence of {len(video_paths)} shots.")
        
        return {
            "job_id": job_id,
            "reference_shot_id": ref_shot_id,
            "creative_specification": creative_spec.model_dump(),
            "research_citations": [c.model_dump() for c in creative_spec.citations],
            "profile_assessments": [a.model_dump() for a in shot_assessments],
            "normalization_results": [n.model_dump() for n in normalization_results],
            "shared_lut_path": shared_lut_path,
            "results": [r.model_dump() for r in graded_results],
            "events": events
        }