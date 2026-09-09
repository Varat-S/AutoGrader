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

def test_api_accepts_all_four_preset_prompts():
    from unittest.mock import patch
    prompts = [
        "Create a warm 800-speed color-negative aesthetic inspired by Portra 800, without attempting exact film-stock emulation. Use soft-to-medium contrast, moderately rich but controlled saturation, gently lifted filmic blacks, warm amber highlights, neutral-to-subtly cool shadows, smooth highlight roll-off, and detailed natural midtones. Preserve each scene’s intended exposure and keep night scenes naturally dark. Avoid a global orange or yellow wash, excessive saturation, crushed shadows, and forced luminance matching between unrelated scenes.",
        "Create a contemporary silver-retention, bleach-bypass-inspired thriller aesthetic. Use firm high contrast, dense neutral blacks, substantially restrained saturation, cool steel-blue shadows, mostly neutral highlights, crisp midtone separation, and controlled highlight roll-off. Preserve subject readability and practical-light detail while maintaining the source scene’s lighting intent. Avoid warm golden highlights, orange coloration, teal-and-orange blockbuster styling, pastel lifted blacks, and excessive colour saturation.",
        "Create a cool, overcast Nordic drama aesthetic. Use soft low contrast, restrained saturation, clean neutral whites and highlights, subtly blue-slate shadows, a gently lifted black toe, broad readable midtones, and delicate highlight compression. Preserve quiet natural lighting and the exposure character of each individual scene. Avoid amber or golden warmth, aggressive teal-and-orange separation, crushed blacks, neon colour, excessive contrast, and global luminance matching across different lighting environments.",
        "Create a controlled neon-noir night aesthetic with cyan-teal shadows and magenta-violet highlights and practical lights. Use moderately strong contrast, deep but readable blacks, vivid colour in illuminated midtones, smooth highlight compression, and restrained saturation in deep shadows. Preserve low-key night exposure and important subject detail; do not brighten night scenes toward daylight. Avoid a global cyan or purple wash, oversaturated blacks, channel clipping, crushed midtones, excessive global saturation, and warm amber dominance."
    ]

    for p in prompts:
        res = client.post("/api/jobs")
        assert res.status_code == 200
        job_id = res.json()["job_id"]
        client.post(f"/api/jobs/{job_id}/load_demo")

        with patch("app.main.run_agent_task"):
            res_run = client.post(
                f"/api/jobs/{job_id}/run",
                json={
                    "creative_prompt": p,
                    "reference_index": 0,
                    "color_profile": "rec709"
                }
            )
            assert res_run.status_code == 200, f"Prompt failed validation: {res_run.text}"
            assert res_run.json()["status"] == "queued"

def test_load_demo_clamps_and_rejects_excess():
    res = client.post("/api/jobs")
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # First load succeeds
    res_load = client.post(f"/api/jobs/{job_id}/load_demo")
    assert res_load.status_code == 200
    data = res_load.json()
    assert len(data["all_clips"]) <= 4

    # Second load with identical clips already present returns 400
    res_load2 = client.post(f"/api/jobs/{job_id}/load_demo")
    assert res_load2.status_code == 400
    assert "no new valid demo clips" in res_load2.json()["detail"].lower() or "limit" in res_load2.json()["detail"].lower()