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
    assert len(res_load.json()["loaded"]) >= 3
    
    # 3. Check Job Status
    res_status = client.get(f"/api/jobs/{job_id}")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["job_id"] == job_id
    assert len(data["source_videos"]) >= 3

def test_api_assess_profiles_and_duplicate_run_protection():
    res = client.post("/api/jobs")
    job_id = res.json()["job_id"]
    
    # Load demo clips
    client.post(f"/api/jobs/{job_id}/load_demo")
    
    # Assess profiles
    res_assess = client.get(f"/api/jobs/{job_id}/assess_profiles")
    assert res_assess.status_code == 200
    assess_data = res_assess.json()
    assert assess_data["status"] == "success"
    assert len(assess_data["assessments"]) >= 3
    for a in assess_data["assessments"]:
        assert "selected_profile" in a
        assert "confidence" in a
        assert "signal_class_hint" in a

    # Run job with background task mocked
    from unittest.mock import patch
    with patch("app.main.run_agent_task"):
        res_run = client.post(
            f"/api/jobs/{job_id}/run",
            json={
                "creative_prompt": "cinematic gold and teal look",
                "reference_index": 0,
                "color_profile": "auto",
                "input_profiles": [
                    {"shot_index": 0, "profile": "rec709"},
                    {"shot_index": 1, "profile": "rec709"}
                ]
            }
        )
        assert res_run.status_code == 200
        assert res_run.json()["status"] == "queued"
        
        # Duplicate run request must be rejected
        res_dup = client.post(
            f"/api/jobs/{job_id}/run",
            json={
                "creative_prompt": "another prompt",
                "reference_index": 0,
                "color_profile": "auto"
            }
        )
        assert res_dup.status_code == 400
        assert "already" in res_dup.json()["detail"].lower()