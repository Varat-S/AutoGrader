let currentJobId = null;
let currentJobResult = null;
let activeShotIdx = 0;
let pollInterval = null;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const clipsList = document.getElementById("clips-list");
const btnLoadDemo = document.getElementById("btn-load-demo");
const btnRun = document.getElementById("btn-run");
const promptInput = document.getElementById("creative-prompt");
const refSelect = document.getElementById("ref-shot-select");

// 1. SETUP & CLIPS
async function ensureJob() {
    if (!currentJobId) {
        const res = await fetch("/api/jobs", { method: "POST" });
        const data = await res.json();
        currentJobId = data.job_id;
    }
    return currentJobId;
}

// Preset chips click handler
document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
        promptInput.value = chip.getAttribute("data-prompt");
    });
});

// Drag & Drop
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleUpload(e.target.files);
    }
});

async function handleUpload(files) {
    const jobId = await ensureJob();
    const formData = new FormData();
    for (let f of files) {
        formData.append("files", f);
    }
    
    clipsList.innerHTML = `<div class="empty-state">Uploading ${files.length} video clips...</div>`;
    const res = await fetch(`/api/jobs/${jobId}/upload`, {
        method: "POST",
        body: formData
    });
    const data = await res.json();
    fetchJobDetails(jobId);
}

btnLoadDemo.addEventListener("click", async () => {
    const jobId = await ensureJob();
    btnLoadDemo.disabled = true;
    btnLoadDemo.innerText = "Loading Demo...";
    
    const res = await fetch(`/api/jobs/${jobId}/load_demo`, { method: "POST" });
    const data = await res.json();
    
    btnLoadDemo.disabled = false;
    btnLoadDemo.innerText = "⚡ Load Demo Sequence (3 Shots)";
    fetchJobDetails(jobId);
});

async function fetchJobDetails(jobId) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    
    if (job.source_videos && job.source_videos.length > 0) {
        clipsList.innerHTML = "";
        refSelect.innerHTML = `<option value="auto">✨ Auto-detect Best Reference Shot (Gemini)</option>`;
        
        job.source_videos.forEach((path, idx) => {
            const name = path.split("/").pop().split("\\").pop();
            const tag = `Shot ${String.fromCharCode(65 + idx)}`;
            
            const row = document.createElement("div");
            row.className = "clip-row";
            row.innerHTML = `
                <span><strong>${tag}</strong>: ${name}</span>
                <span class="clip-tag">${idx === 0 ? "Shot A" : "Shot " + String.fromCharCode(65 + idx)}</span>
            `;
            clipsList.appendChild(row);
            
            const opt = document.createElement("option");
            opt.value = idx;
            opt.innerText = `${tag}: ${name}`;
            refSelect.appendChild(opt);
        });
        
        btnRun.disabled = false;
    }
}

// 2. RUN WORKFLOW
btnRun.addEventListener("click", async () => {
    if (!currentJobId) return;
    
    btnRun.disabled = true;
    document.getElementById("agent-activity-panel").style.display = "block";
    document.getElementById("results-panel").style.display = "none";
    
    const refVal = refSelect.value === "auto" ? null : parseInt(refSelect.value);
    const colorProfileVal = document.getElementById("color-profile-select").value;
    
    await fetch(`/api/jobs/${currentJobId}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            creative_prompt: promptInput.value,
            reference_index: refVal,
            color_profile: colorProfileVal
        })
    });
    
    startPolling(currentJobId);
});

function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    
    pollInterval = setInterval(async () => {
        const res = await fetch(`/api/jobs/${jobId}`);
        const job = await res.json();
        
        updateAgentUI(job);
        
        if (job.state === "completed") {
            clearInterval(pollInterval);
            currentJobResult = job.result;
            renderResults(job.result, job.source_videos);
        } else if (job.state === "failed") {
            clearInterval(pollInterval);
            alert("Error during agent processing: " + (job.error || "Unknown error"));
        }
    }, 1000);
}

function updateAgentUI(job) {
    document.getElementById("progress-fill").style.width = `${job.progress}%`;
    
    // Log feed
    const logFeed = document.getElementById("log-feed");
    logFeed.innerHTML = "";
    job.events.forEach(ev => {
        const div = document.createElement("div");
        div.className = "log-line";
        if (ev.includes("Gemini")) div.classList.add("gemini");
        if (ev.includes("Parallel")) div.classList.add("parallel");
        if (ev.includes("CIELAB") || ev.includes("Measuring")) div.classList.add("cv");
        if (ev.includes("revision") || ev.includes("Revised")) div.classList.add("revise");
        div.innerText = `> ${ev}`;
        logFeed.appendChild(div);
    });
    logFeed.scrollTop = logFeed.scrollHeight;
    
    // Stepper states
    const steps = [
        { id: "step-perceive", threshold: 20 },
        { id: "step-research", threshold: 40 },
        { id: "step-measure", threshold: 55 },
        { id: "step-grade", threshold: 75 },
        { id: "step-evaluate", threshold: 90 },
        { id: "step-deliver", threshold: 100 }
    ];
    
    steps.forEach((s, idx) => {
        const el = document.getElementById(s.id);
        if (job.progress >= s.threshold) {
            el.className = "step-item done";
        } else if (idx === 0 || job.progress >= steps[idx - 1].threshold) {
            el.className = "step-item active";
        } else {
            el.className = "step-item";
        }
    });
}

// 3. RENDER RESULTS
function renderResults(result, sourceVideos) {
    document.getElementById("results-panel").style.display = "block";
    document.getElementById("agent-status-badge").innerText = "Completed";
    document.getElementById("agent-status-badge").className = "status-badge badge-done";
    
    // Creative Spec Card
    const spec = result.creative_specification;
    const specCard = document.getElementById("creative-spec-card");
    specCard.innerHTML = `
        <div class="spec-title-row">
            <div class="spec-title">✨ Look: "${spec.look_title}" (Reference Shot: ${result.reference_shot_id})</div>
        </div>
        <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">${spec.target_aesthetic}</p>
        <div class="spec-badges">
            <span class="spec-badge">Contrast: ${spec.contrast_intent}x</span>
            <span class="spec-badge">Saturation: ${spec.saturation_intent}x</span>
            <span class="spec-badge">Highlights: ${spec.highlight_bias}</span>
            <span class="spec-badge">Shadows: ${spec.shadow_bias}</span>
            <span class="spec-badge">Skin Rendering: ${spec.skin_rendering_intent}</span>
        </div>
    `;
    
    // Citations
    const citationsList = document.getElementById("citations-list");
    citationsList.innerHTML = "";
    result.research_citations.forEach(c => {
        const link = document.createElement("a");
        link.className = "citation-link";
        link.href = c.url || "#";
        link.target = "_blank";
        link.innerText = `🔗 ${c.title}`;
        citationsList.appendChild(link);
    });
    
    // Shot Tabs
    const shotTabs = document.getElementById("shot-tabs");
    shotTabs.innerHTML = "";
    result.results.forEach((r, idx) => {
        const isMaster = (r.target_shot_id === result.reference_shot_id);
        const btn = document.createElement("button");
        btn.className = `shot-tab ${idx === 0 ? "active" : ""}`;
        btn.innerText = `${r.target_shot_id}${isMaster ? " (Master Ref)" : ""}`;
        btn.addEventListener("click", () => {
            document.querySelectorAll(".shot-tab").forEach(t => t.classList.remove("active"));
            btn.classList.add("active");
            displayShotResult(r, idx, sourceVideos);
        });
        shotTabs.appendChild(btn);
    });
    
    if (result.results.length > 0) {
        displayShotResult(result.results[0], 0, sourceVideos);
    }
}

function displayShotResult(res, shotIdx, sourceVideos) {
    const jobId = currentJobId;
    
    // Source Video Path
    let sourceFilename = "source.mp4";
    if (sourceVideos && sourceVideos[shotIdx]) {
        sourceFilename = sourceVideos[shotIdx].split("/").pop().split("\\").pop();
    }
    
    const beforeVideoUrl = `/api/jobs/${jobId}/files/${sourceFilename}`;
    const gradedVideoFilename = res.output_video_path.split("/").pop().split("\\").pop();
    const lutFilename = res.lut_path.split("/").pop().split("\\").pop();
    
    const afterVideoUrl = `/api/jobs/${jobId}/files/${gradedVideoFilename}`;
    const lutUrl = `/api/jobs/${jobId}/files/${lutFilename}`;
    
    const playerBefore = document.getElementById("player-before");
    const playerAfter = document.getElementById("player-after");
    
    playerBefore.src = beforeVideoUrl;
    playerAfter.src = afterVideoUrl;
    
    document.getElementById("label-source-shot").innerText = `Source: ${res.target_shot_id} (${sourceFilename})`;
    document.getElementById("label-graded-shot").innerText = `Graded: ${res.target_shot_id}`;
    
    // Scores
    const beforeScore = Math.round(res.before_consistency.overall_score);
    const afterScore = Math.round(res.after_consistency.overall_score);
    document.getElementById("score-before").innerText = beforeScore;
    document.getElementById("score-after").innerText = afterScore;
    
    const delta = afterScore - beforeScore;
    document.getElementById("score-delta").innerText = delta >= 0 ? `+${delta} points consistency match` : `Master Style Applied`;
    
    document.getElementById("metric-lum").innerText = `${Math.round(res.after_consistency.luminance_similarity)} / 100`;
    document.getElementById("metric-dist").innerText = `${Math.round(res.after_consistency.color_distribution_similarity)} / 100`;
    
    document.getElementById("explanation-text").innerText = res.explanation;
    
    // Download Buttons with explicit shot tags
    const btnVid = document.getElementById("btn-download-video");
    btnVid.href = afterVideoUrl;
    btnVid.download = `${res.target_shot_id}_graded.mp4`;
    btnVid.innerText = `⬇ Download ${res.target_shot_id} Video (.mp4)`;
    
    const btnLut = document.getElementById("btn-download-lut");
    btnLut.href = lutUrl;
    btnLut.download = `${res.target_shot_id}_grade.cube`;
    btnLut.innerText = `⬇ Download ${res.target_shot_id} 3D LUT (.cube)`;
}