import os
import shutil
import uuid
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.agent import AutonomousColoristAgent
from app.media.ffmpeg import probe_video
from app.tools.measure_color import measure_shot_color
from app.tools.calculate_grade import assess_input_profile
from app.models.grade import InputProfile, ShotProfileSelection

app = FastAPI(
    title="AutoGrader — Autonomous Multimodal Cinema Colorist",
    description="Multimodal Autonomous Colorist using Gemini Multimodal Vision, Parallel Cinematography Search, and Staged 32-bit Floating-Point DI Color Science.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("output/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry and concurrency semaphore
jobs: Dict[str, Dict[str, Any]] = {}
MAX_CONCURRENT_JOBS = 3
active_job_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024 # 500 MB
MAX_CLIPS_PER_JOB = 4

class RunJobRequest(BaseModel):
    creative_prompt: str = Field(..., max_length=1000, description="Filmmaker aesthetic description (max 1000 characters)")
    reference_index: Optional[int] = Field(None, ge=0, le=3, description="Optional 0-indexed reference clip selection")
    color_profile: str = Field("auto", description="'auto', 'rec709', 'sony_slog3_sgamut3cine', 'apple_log_rec2020', 'dji_dlog_dgamut', 'dji_dlog_m_rec709', 'generic_log_experimental'")
    input_profiles: Optional[List[ShotProfileSelection]] = Field(None, description="Per-shot typed profile selections")

def run_agent_task(
    job_id: str,
    prompt: str,
    ref_idx: Optional[int],
    color_profile: str = "auto",
    input_profiles: Optional[List[ShotProfileSelection]] = None
):
    job = jobs.get(job_id)
    if not job:
        return
        
    with active_job_semaphore:
        job["state"] = "running"
        job["progress"] = 10
        
        job_dir = JOBS_DIR / job_id
        agent = AutonomousColoristAgent(work_dir=str(job_dir / "output"))
        
        def on_progress(event_msg: str):
            job["events"].append(event_msg)
            if "Inspecting" in event_msg:
                job["progress"] = 25
            elif "Researching" in event_msg:
                job["progress"] = 45
            elif "Synthesized" in event_msg:
                job["progress"] = 60
            elif "Rendering" in event_msg or "Evaluating" in event_msg:
                job["progress"] = min(92, job["progress"] + 8)
            elif "complete" in event_msg:
                job["progress"] = 100
                
        try:
            source_paths = job["source_videos"]
            result = agent.process_sequence(
                video_paths=source_paths,
                creative_prompt=prompt,
                reference_index=ref_idx,
                color_profile=color_profile,
                input_profiles=input_profiles,
                job_id=job_id,
                progress_callback=on_progress
            )
            job["state"] = "completed"
            job["progress"] = 100
            job["result"] = result
        except Exception as e:
            job["state"] = "failed"
            job["error"] = str(e)
            job["events"].append(f"Error during execution: {str(e)}")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "AutoGrader Autonomous Cinema Colorist"}

@app.post("/api/jobs")
def create_job():
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    (job_dir / "source").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)
    
    jobs[job_id] = {
        "job_id": job_id,
        "state": "created",
        "progress": 0,
        "source_videos": [],
        "events": ["Job created. Ready for video upload."],
        "result": None,
        "error": None
    }
    return {"job_id": job_id}

@app.post("/api/jobs/{job_id}/upload")
async def upload_videos(job_id: str, files: List[UploadFile] = File(...)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if len(job["source_videos"]) + len(files) > MAX_CLIPS_PER_JOB:
        raise HTTPException(status_code=400, detail=f"Maximum of {MAX_CLIPS_PER_JOB} video clips allowed per job.")
        
    job_source_dir = JOBS_DIR / job_id / "source"
    uploaded_paths = []
    
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")
            
        safe_filename = Path(f.filename).name
        shot_num = len(job["source_videos"]) + 1
        dest_filename = f"shot_{shot_num}_{safe_filename}"
        dest_path = job_source_dir / dest_filename
        
        # Save file with size enforcement
        total_bytes = 0
        with open(dest_path, "wb") as buffer:
            while chunk := await f.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File '{safe_filename}' exceeds 500MB limit.")
                buffer.write(chunk)
                
        # Validate video decoding and metadata with ffprobe
        try:
            info = probe_video(str(dest_path))
            if info["width"] <= 0 or info["height"] <= 0 or info["duration_sec"] <= 0:
                dest_path.unlink(missing_ok=True)
                raise ValueError("Invalid video stream dimensions or duration.")
        except Exception as e:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Video validation failed for '{safe_filename}': {str(e)}")
            
        dest_str = str(dest_path)
        if dest_str not in job["source_videos"]:
            job["source_videos"].append(dest_str)
        uploaded_paths.append(dest_str)
        
    job["events"].append(f"Uploaded and validated {len(files)} clip(s).")
    return {
        "status": "success",
        "uploaded": [Path(p).name for p in uploaded_paths],
        "all_clips": [Path(p).name for p in job["source_videos"]],
        "total_clips": len(job["source_videos"])
    }

@app.post("/api/jobs/{job_id}/load_demo")
def load_demo_sequence(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    demo_dir = Path("tests/fixtures/demo_sequence")
    if demo_dir.exists() and list(demo_dir.glob("*.mp4")):
        fixtures_dir = demo_dir
    else:
        fixtures_dir = Path("tests/fixtures/sample_videos")

    if not fixtures_dir.exists():
        raise HTTPException(status_code=400, detail="Demo fixtures directory not found")
        
    job = jobs[job_id]
    if len(job["source_videos"]) >= MAX_CLIPS_PER_JOB:
        raise HTTPException(status_code=400, detail=f"Job already reached maximum limit of {MAX_CLIPS_PER_JOB} video clips.")

    available_samples = [s for s in sorted(fixtures_dir.glob("*.mp4")) if s.is_file() and s.stat().st_size > 0]
    if not available_samples:
        raise HTTPException(status_code=400, detail="No valid demo video clips found.")

    remaining_slots = MAX_CLIPS_PER_JOB - len(job["source_videos"])
    samples_to_load = available_samples[:remaining_slots]

    job_source_dir = JOBS_DIR / job_id / "source"
    loaded = []
    
    # Avoid duplicate content by tracking existing basenames
    existing_basenames = {Path(p).name.split("_", 2)[-1] for p in job["source_videos"]}
    
    for sample in samples_to_load:
        if sample.name in existing_basenames:
            continue
        idx = len(job["source_videos"]) + 1
        dest_filename = f"shot_{idx}_{sample.name}"
        dest = job_source_dir / dest_filename
        shutil.copyfile(sample, dest)
        dest_str = str(dest)
        if dest_str not in job["source_videos"]:
            job["source_videos"].append(dest_str)
            existing_basenames.add(sample.name)
            loaded.append(dest_filename)
        
    if not loaded:
        raise HTTPException(status_code=400, detail="No new valid demo clips could be loaded (all clips already present or limit reached).")

    job["events"].append(f"Loaded {len(loaded)} demo clip(s).")
    return {"status": "success", "loaded": loaded, "all_clips": [Path(p).name for p in job["source_videos"]]}

@app.get("/api/jobs/{job_id}/assess_profiles")
def assess_job_profiles(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    assessments = []
    for i, path in enumerate(job["source_videos"]):
        shot_id = f"shot_{chr(65 + i)}"
        try:
            info = probe_video(path)
            metrics = measure_shot_color(path, shot_id=shot_id)
            assessment = assess_input_profile(shot_id, info, metrics, requested_profile="auto_ask", shot_index=i, user_confirmed=False)
            assessments.append(assessment.model_dump())
        except Exception as e:
            assessments.append({
                "shot_index": i,
                "shot_id": shot_id,
                "requested_profile": "auto_ask",
                "selected_profile": "rec709",
                "metadata_recommendation": None,
                "metadata_hint": "error",
                "signal_class_hint": "display_ready",
                "recommended_profile": "rec709",
                "resolved_profile": "rec709",
                "resolution_source": "fallback_default",
                "requires_confirmation": False,
                "user_confirmed": False,
                "confidence": 0.0,
                "reasons": [f"Failed to assess profile: {str(e)}"],
                "profile_mismatch_warning": False,
                "warning_message": ""
            })
            
    return {"status": "success", "assessments": assessments}

@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str, request: RunJobRequest, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job.get("state") in ["queued", "running"]:
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is already {job['state']}. Duplicate runs are not permitted.")
        
    if len(job["source_videos"]) == 0:
        raise HTTPException(status_code=400, detail="No source video clips uploaded")
        
    if request.reference_index is not None:
        if request.reference_index < 0 or request.reference_index >= len(job["source_videos"]):
            raise HTTPException(status_code=400, detail="Invalid reference_index")
            
    # Normalize sequence color profile
    seq_prof = request.color_profile.lower().strip()
    if seq_prof == "apple_log_apple_wide_gamut":
        seq_prof = "apple_log_rec2020"
    elif seq_prof in ["rec.709", "bt709"]:
        seq_prof = "rec709"
    elif seq_prof in ["dji_dlog_m", "dlog_m", "dlog-m", "dji-dlog-m", "dji_dlog_m_rec709"]:
        seq_prof = "dji_dlog_m_rec709"
    elif seq_prof in ["dji_dlog", "dlog", "dji"]:
        seq_prof = "dji_dlog_dgamut"
        
    valid_sequence_profiles = {
        "auto", "rec709", "sony_slog3_sgamut3cine", "apple_log_rec2020", "dji_dlog_dgamut", "dji_dlog_m_rec709", "generic_log_experimental"
    }
    if seq_prof not in valid_sequence_profiles:
        raise HTTPException(status_code=400, detail=f"Invalid color_profile '{request.color_profile}'. Valid options: {sorted(list(valid_sequence_profiles))}")
    request.color_profile = seq_prof

    num_videos = len(job["source_videos"])
    
    provided_profiles: Dict[int, ShotProfileSelection] = {}
    if request.input_profiles is not None:
        seen_indices = set()
        for p_sel in request.input_profiles:
            if p_sel.shot_index < 0 or p_sel.shot_index >= num_videos:
                raise HTTPException(
                    status_code=400,
                    detail=f"shot_index {p_sel.shot_index} is out of range for sequence of length {num_videos}."
                )
            if p_sel.shot_index in seen_indices:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate shot_index {p_sel.shot_index} in input_profiles."
                )
            seen_indices.add(p_sel.shot_index)
            provided_profiles[p_sel.shot_index] = p_sel

    final_input_profiles: List[ShotProfileSelection] = []
    unresolved_shots = []

    for i, path in enumerate(job["source_videos"]):
        shot_id = f"shot_{chr(65 + i)}"
        if i in provided_profiles:
            p_sel = provided_profiles[i]
            if p_sel.profile == InputProfile.AUTO_ASK or p_sel.profile.value == "auto_ask":
                unresolved_shots.append(shot_id)
            else:
                final_input_profiles.append(p_sel)
        else:
            # Fall back to sequence default
            if request.color_profile == "auto":
                info = probe_video(path)
                metrics = measure_shot_color(path, shot_id=shot_id)
                assessment = assess_input_profile(shot_id, info, metrics, requested_profile="auto", shot_index=i, user_confirmed=False)
                if assessment.requires_confirmation or assessment.resolved_profile is None or assessment.resolved_profile == "auto_ask":
                    unresolved_shots.append(shot_id)
                else:
                    final_input_profiles.append(ShotProfileSelection(
                        shot_index=i,
                        profile=InputProfile(assessment.resolved_profile),
                        user_confirmed=True
                    ))
            elif request.color_profile == "auto_ask":
                unresolved_shots.append(shot_id)
            else:
                final_input_profiles.append(ShotProfileSelection(
                    shot_index=i,
                    profile=InputProfile(request.color_profile),
                    user_confirmed=True
                ))

    if unresolved_shots:
        raise HTTPException(
            status_code=409,
            detail=f"Camera profile confirmation required before running. Unresolved shots: {unresolved_shots}"
        )

    request.input_profiles = final_input_profiles
        
    job["state"] = "queued"
    background_tasks.add_task(
        run_agent_task,
        job_id=job_id,
        prompt=request.creative_prompt,
        ref_idx=request.reference_index,
        color_profile=request.color_profile,
        input_profiles=request.input_profiles
    )
    
    return {"status": "queued", "job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/api/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str):
    safe_name = Path(filename).name
    job_dir = JOBS_DIR / job_id
    
    cand1 = job_dir / "source" / safe_name
    cand2 = job_dir / "output" / safe_name
    
    if cand1.exists():
        return FileResponse(str(cand1))
    elif cand2.exists():
        return FileResponse(str(cand2))
    else:
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found for job {job_id}")

# Mount static frontend
STATIC_DIR = Path("app/static")
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")