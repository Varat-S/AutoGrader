import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_job_lifecycle():
    # 1. Create Job
    res = client.post("/api/jobs")
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert job_id.startswith("job_")
    
    # 2. Load Demo Clips
    res_load = client.post(f"/api/jobs/{job_id}/load_demo")
    assert res_load.status_code == 200
    assert len(res_load.json()["loaded"]) == 3
    
    # 3. Check Job Status
    res_status = client.get(f"/api/jobs/{job_id}")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["job_id"] == job_id
    assert len(data["source_videos"]) == 3