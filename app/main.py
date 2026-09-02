import os
import shutil
import uuid
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.agent import AutonomousColoristAgent
from app.media.ffmpeg import probe_video

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
    creative_prompt: str = Field(..., max_length=500, description="Filmmaker aesthetic description (max 500 characters)")
    reference_index: Optional[int] = Field(None, ge=0, le=3, description="Optional 0-indexed reference clip selection")
    color_profile: str = Field("auto", description="'auto', 'Rec.709', 'Log', or 'Generic Log'")

def run_agent_task(job_id: str, prompt: str, ref_idx: Optional[int], color_profile: str = "auto"):
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
        dest_path = job_source_dir / safe_filename
        
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
    return {"status": "success", "uploaded": [Path(p).name for p in uploaded_paths], "total_clips": len(job["source_videos"])}

@app.post("/api/jobs/{job_id}/load_demo")
def load_demo_sequence(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    fixtures_dir = Path("tests/fixtures/sample_videos")
    if not fixtures_dir.exists():
        raise HTTPException(status_code=500, detail="Demo fixtures not found")
        
    job = jobs[job_id]
    job_source_dir = JOBS_DIR / job_id / "source"
    
    loaded = []
    for sample in sorted(fixtures_dir.glob("*.mp4")):
        dest = job_source_dir / sample.name
        shutil.copyfile(sample, dest)
        dest_str = str(dest)
        if dest_str not in job["source_videos"]:
            job["source_videos"].append(dest_str)
        loaded.append(sample.name)
        
    job["events"].append(f"Loaded {len(loaded)} benchmark demo clip(s).")
    return {"status": "success", "loaded": loaded, "all_clips": [Path(p).name for p in job["source_videos"]]}

@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str, request: RunJobRequest, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if len(job["source_videos"]) == 0:
        raise HTTPException(status_code=400, detail="No source video clips uploaded")
        
    if request.reference_index is not None:
        if request.reference_index < 0 or request.reference_index >= len(job["source_videos"]):
            raise HTTPException(status_code=400, detail="Invalid reference_index")
            
    valid_profiles = {"auto", "Rec.709", "Log", "Generic Log", "dlog", "slog"}
    if request.color_profile not in valid_profiles:
        raise HTTPException(status_code=400, detail=f"Invalid color_profile '{request.color_profile}'. Valid options: {valid_profiles}")
        
    background_tasks.add_task(
        run_agent_task,
        job_id=job_id,
        prompt=request.creative_prompt,
        ref_idx=request.reference_index,
        color_profile=request.color_profile
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