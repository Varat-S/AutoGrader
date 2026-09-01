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
from pydantic import BaseModel

from app.agent import AutonomousColoristAgent

app = FastAPI(
    title="Autonomous Multimodal Colorist Assistant",
    description="Research-to-Grade Autonomous Colorist using Google ADK, Gemini 3.6 Flash, Parallel Search, and OpenCV/FFmpeg.",
    version="1.0.0"
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

# In-memory and persisted job registry
jobs: Dict[str, Dict[str, Any]] = {}

class RunJobRequest(BaseModel):
    creative_prompt: str
    reference_index: Optional[int] = None
    color_profile: str = "auto"

def run_agent_task(job_id: str, prompt: str, ref_idx: Optional[int], color_profile: str = "auto"):
    job = jobs[job_id]
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
        elif "Calculating" in event_msg or "Rendering" in event_msg:
            job["progress"] = min(90, job["progress"] + 10)
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
    return {"status": "healthy", "service": "Autonomous Multimodal Colorist Assistant"}

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
    job_source_dir = JOBS_DIR / job_id / "source"
    
    uploaded_paths = []
    for f in files:
        safe_filename = Path(f.filename).name
        dest_path = job_source_dir / safe_filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        dest_str = str(dest_path)
        if dest_str not in job["source_videos"]:
            job["source_videos"].append(dest_str)
        uploaded_paths.append(dest_str)
        
    all_filenames = [Path(p).name for p in job["source_videos"]]
    job["events"].append(f"Uploaded {len(files)} video clips: {', '.join([Path(p).name for p in uploaded_paths])}.")
    return {
        "status": "success",
        "uploaded": [Path(p).name for p in uploaded_paths],
        "all_clips": all_filenames,
        "total_videos": len(all_filenames)
    }

@app.post("/api/jobs/{job_id}/load_demo")
def load_demo_footage(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    fixtures_dir = Path("tests/fixtures/sample_videos")
    demo_files = ["neutral_reference.mp4", "underexposed.mp4", "warm_cast.mp4"]
    
    job = jobs[job_id]
    job_source_dir = JOBS_DIR / job_id / "source"
    
    loaded_paths = []
    for fname in demo_files:
        src = fixtures_dir / fname
        if src.exists():
            dst = job_source_dir / fname
            shutil.copy(src, dst)
            loaded_paths.append(str(dst))
            
    job["source_videos"] = loaded_paths
    job["events"].append("Loaded 3 demo video clips (Reference, Underexposed, Warm Cast).")
    all_filenames = [Path(p).name for p in loaded_paths]
    return {"status": "success", "loaded": all_filenames, "all_clips": all_filenames}

@app.post("/api/jobs/{job_id}/run")
def start_job(job_id: str, req: RunJobRequest, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if not job["source_videos"]:
        raise HTTPException(status_code=400, detail="No video files uploaded yet")
        
    if job["state"] == "running":
        raise HTTPException(status_code=400, detail="Job is already running")
        
    job["events"].append(f"Starting grading workflow with creative prompt: '{req.creative_prompt}'")
    background_tasks.add_task(run_agent_task, job_id, req.creative_prompt, req.reference_index, req.color_profile)
    return {"status": "started", "job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/api/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str):
    job_dir = JOBS_DIR / job_id
    
    # Check output folder first, then source folder
    file_path = job_dir / "output" / filename
    if not file_path.exists():
        file_path = job_dir / "source" / filename
        
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
    media_type = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)

# Mount frontend static files
static_dir = Path("app/static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)