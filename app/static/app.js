// AutoGrader — Autonomous Multimodal Colorist Studio

let currentJobId = null;
let pollInterval = null;
let isDraggingSlider = false;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const btnLoadDemo = document.getElementById("btn-load-demo");
const clipsList = document.getElementById("clips-list");
const refSelect = document.getElementById("ref-shot-select");
const colorSelect = document.getElementById("color-profile-select");
const promptInput = document.getElementById("creative-prompt");
const btnRun = document.getElementById("btn-run");
const promptChips = document.querySelectorAll(".chip");

// Split Slider DOM Elements
const sliderFrame = document.getElementById("slider-frame");
const sliderDivider = document.getElementById("split-slider-divider");
const sliderAfterClip = document.getElementById("slider-after-clip");
const playerSliderBefore = document.getElementById("player-slider-before");
const playerSliderAfter = document.getElementById("player-slider-after");
const btnSliderPlay = document.getElementById("btn-slider-play");
const iconPlay = document.getElementById("icon-play");
const iconPause = document.getElementById("icon-pause");
const sliderTimeline = document.getElementById("slider-timeline");
const sliderTimeDisplay = document.getElementById("slider-time-display");

// View Mode DOM Elements
const btnViewSlider = document.getElementById("btn-view-slider");
const btnViewSide = document.getElementById("btn-view-side");
const comparisonSliderView = document.getElementById("comparison-slider-view");
const sideBySideView = document.getElementById("side-by-side-view");

// INITIALIZE APP
document.addEventListener("DOMContentLoaded", () => {
    initJob();
    setupEventListeners();
    initComparisonSlider();
    initViewModeToggle();
});

async function initJob() {
    try {
        const res = await fetch("/api/jobs", { method: "POST" });
        const data = await res.json();
        currentJobId = data.job_id;
        console.log("Initialized job:", currentJobId);
    } catch (e) {
        console.error("Failed to initialize job:", e);
    }
}

function setupEventListeners() {
    // Dropzone
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) uploadFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) uploadFiles(fileInput.files);
    });

    // Load Demo Footage
    btnLoadDemo.addEventListener("click", async () => {
        if (!currentJobId) await initJob();
        btnLoadDemo.disabled = true;
        btnLoadDemo.innerText = "Loading demo clips...";
        try {
            const res = await fetch(`/api/jobs/${currentJobId}/load_demo`, { method: "POST" });
            const data = await res.json();
            updateClipsList(data.loaded);
            btnRun.disabled = false;
        } catch (e) {
            console.error("Demo load failed:", e);
        } finally {
            btnLoadDemo.disabled = false;
            btnLoadDemo.innerText = "Load Demo Sequence (3 Shots)";
        }
    });

    // Aesthetic Presets
    promptChips.forEach(chip => {
        chip.addEventListener("click", () => {
            promptInput.value = chip.getAttribute("data-prompt");
            promptInput.focus();
        });
    });

    // Run Workflow
    btnRun.addEventListener("click", async () => {
        if (!currentJobId) return;
        btnRun.disabled = true;
        document.getElementById("agent-activity-panel").style.display = "block";
        document.getElementById("results-panel").style.display = "none";
        
        const refVal = refSelect.value === "auto" ? null : parseInt(refSelect.value);
        const colorProfileVal = colorSelect.value;
        
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
}

// FILE UPLOAD
async function uploadFiles(files) {
    if (!currentJobId) await initJob();
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }
    
    try {
        const res = await fetch(`/api/jobs/${currentJobId}/upload`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        const clips = data.all_clips || data.loaded || (Array.isArray(data.uploaded) ? data.uploaded : []);
        updateClipsList(clips);
        btnRun.disabled = (clips.length === 0);
    } catch (e) {
        console.error("Upload error:", e);
    }
}

function updateClipsList(filenames) {
    if (!Array.isArray(filenames)) {
        console.warn("Expected array of filenames, got:", filenames);
        filenames = [];
    }
    
    clipsList.innerHTML = "";
    refSelect.innerHTML = '<option value="auto" selected>Auto-detect Optimal Reference Shot (Gemini)</option>';
    
    if (filenames.length === 0) {
        clipsList.innerHTML = '<div class="empty-state">No clips loaded yet. Drop files above or load the demo sequence.</div>';
        btnRun.disabled = true;
        return;
    }
    
    filenames.forEach((fname, idx) => {
        const shotId = `shot_${String.fromCharCode(65 + idx)}`;
        const row = document.createElement("div");
        row.className = "clip-row";
        row.innerHTML = `
            <span><strong>${shotId}</strong>: ${fname}</span>
            <span class="clip-tag">${fname.endsWith(".MP4") || fname.endsWith(".mp4") ? "MP4" : "MOV"}</span>
        `;
        clipsList.appendChild(row);
        
        const opt = document.createElement("option");
        opt.value = idx;
        opt.innerText = `Shot ${String.fromCharCode(65 + idx)} (${fname})`;
        refSelect.appendChild(opt);
    });
    
    btnRun.disabled = false;
}

// POLLING & ACTIVITY FEED
function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    
    pollInterval = setInterval(async () => {
        const res = await fetch(`/api/jobs/${jobId}`);
        const job = await res.json();
        
        updateActivityFeed(job);
        
        if (job.state === "completed") {
            clearInterval(pollInterval);
            btnRun.disabled = false;
            renderResults(job.result, job.source_videos);
        } else if (job.state === "failed") {
            clearInterval(pollInterval);
            btnRun.disabled = false;
            document.getElementById("agent-status-badge").innerText = "Failed";
            document.getElementById("agent-status-badge").className = "status-badge badge-running";
        }
    }, 1500);
}

function updateActivityFeed(job) {
    document.getElementById("progress-fill").style.width = `${job.progress}%`;
    const logFeed = document.getElementById("log-feed");
    logFeed.innerHTML = "";
    
    job.events.forEach(evt => {
        const line = document.createElement("div");
        line.className = "log-line";
        if (evt.includes("Gemini")) line.classList.add("gemini");
        else if (evt.includes("Parallel")) line.classList.add("parallel");
        else if (evt.includes("CIELAB") || evt.includes("LUT")) line.classList.add("cv");
        else if (evt.includes("Revise") || evt.includes("evaluate")) line.classList.add("revise");
        line.innerText = evt;
        logFeed.appendChild(line);
    });
    logFeed.scrollTop = logFeed.scrollHeight;

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

// -------------------------------------------------------------
// INTERACTIVE SPLIT SLIDER & SYNCHRONIZED PLAYBACK ENGINE
// -------------------------------------------------------------
function initComparisonSlider() {
    function setSliderPosition(clientX) {
        if (!sliderFrame) return;
        const rect = sliderFrame.getBoundingClientRect();
        let posX = clientX - rect.left;
        posX = Math.max(0, Math.min(posX, rect.width));
        const pct = (posX / rect.width) * 100;
        
        sliderDivider.style.left = `${pct}%`;
        sliderAfterClip.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
    }

    // Mouse Events
    sliderFrame.addEventListener("mousedown", (e) => {
        isDraggingSlider = true;
        sliderFrame.classList.add("dragging");
        setSliderPosition(e.clientX);
    });

    window.addEventListener("mousemove", (e) => {
        if (isDraggingSlider) setSliderPosition(e.clientX);
    });

    window.addEventListener("mouseup", () => {
        if (isDraggingSlider) {
            isDraggingSlider = false;
            sliderFrame.classList.remove("dragging");
        }
    });

    // Touch Events
    sliderFrame.addEventListener("touchstart", (e) => {
        if (e.touches.length > 0) {
            isDraggingSlider = true;
            sliderFrame.classList.add("dragging");
            setSliderPosition(e.touches[0].clientX);
        }
    }, { passive: true });

    window.addEventListener("touchmove", (e) => {
        if (isDraggingSlider && e.touches.length > 0) {
            setSliderPosition(e.touches[0].clientX);
        }
    }, { passive: true });

    window.addEventListener("touchend", () => {
        if (isDraggingSlider) {
            isDraggingSlider = false;
            sliderFrame.classList.remove("dragging");
        }
    });

    // Synchronized Video Playback Controls
    btnSliderPlay.addEventListener("click", toggleSliderPlayback);
    sliderFrame.addEventListener("click", (e) => {
        // Toggle play if click was not a drag
        if (!isDraggingSlider) toggleSliderPlayback();
    });

    function toggleSliderPlayback(e) {
        if (e) e.stopPropagation();
        if (playerSliderBefore.paused) {
            playerSliderBefore.play();
            playerSliderAfter.play();
            iconPlay.style.display = "none";
            iconPause.style.display = "block";
        } else {
            playerSliderBefore.pause();
            playerSliderAfter.pause();
            iconPlay.style.display = "block";
            iconPause.style.display = "none";
        }
    }

    // Scrubbing Timeline
    playerSliderBefore.addEventListener("timeupdate", () => {
        if (playerSliderBefore.duration) {
            const pct = (playerSliderBefore.currentTime / playerSliderBefore.duration) * 100;
            sliderTimeline.value = pct;
            sliderTimeDisplay.innerText = `${formatTime(playerSliderBefore.currentTime)} / ${formatTime(playerSliderBefore.duration)}`;
            
            // Keep after video frame-locked in sync
            if (Math.abs(playerSliderAfter.currentTime - playerSliderBefore.currentTime) > 0.08) {
                playerSliderAfter.currentTime = playerSliderBefore.currentTime;
            }
        }
    });

    sliderTimeline.addEventListener("input", () => {
        if (playerSliderBefore.duration) {
            const targetTime = (sliderTimeline.value / 100) * playerSliderBefore.duration;
            playerSliderBefore.currentTime = targetTime;
            playerSliderAfter.currentTime = targetTime;
        }
    });

    // Keyboard Space shortcut
    window.addEventListener("keydown", (e) => {
        if (e.code === "Space" && document.activeElement.tagName !== "TEXTAREA" && document.activeElement.tagName !== "INPUT") {
            e.preventDefault();
            toggleSliderPlayback();
        }
    });
}

function formatTime(seconds) {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}

// -------------------------------------------------------------
// VIEW MODE TOGGLE (SPLIT SLIDER vs SIDE BY SIDE)
// -------------------------------------------------------------
function initViewModeToggle() {
    btnViewSlider.addEventListener("click", () => {
        btnViewSlider.classList.add("active");
        btnViewSide.classList.remove("active");
        comparisonSliderView.style.display = "block";
        sideBySideView.style.display = "none";
    });

    btnViewSide.addEventListener("click", () => {
        btnViewSide.classList.add("active");
        btnViewSlider.classList.remove("active");
        sideBySideView.style.display = "grid";
        comparisonSliderView.style.display = "none";
    });
}

// -------------------------------------------------------------
// RENDER RESULTS DASHBOARD
// -------------------------------------------------------------
function renderResults(result, sourceVideos) {
    document.getElementById("results-panel").style.display = "block";
    document.getElementById("agent-status-badge").innerText = "Completed";
    document.getElementById("agent-status-badge").className = "status-badge badge-done";
    
    // Creative Spec Card
    const spec = result.creative_specification;
    const specCard = document.getElementById("creative-spec-card");
    specCard.innerHTML = `
        <div class="spec-title-row">
            <div class="spec-title">Look: "${spec.look_title}" (Reference Shot: ${result.reference_shot_id})</div>
        </div>
        <p style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem;">${spec.target_aesthetic}</p>
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
        link.innerText = `Source: ${c.title}`;
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
    
    let sourceFilename = "source.mp4";
    if (sourceVideos && sourceVideos[shotIdx]) {
        sourceFilename = sourceVideos[shotIdx].split("/").pop().split("\\").pop();
    }
    
    const beforeVideoUrl = `/api/jobs/${jobId}/files/${sourceFilename}`;
    const gradedVideoFilename = res.output_video_path.split("/").pop().split("\\").pop();
    const lutFilename = res.lut_path.split("/").pop().split("\\").pop();
    
    const afterVideoUrl = `/api/jobs/${jobId}/files/${gradedVideoFilename}`;
    const lutUrl = `/api/jobs/${jobId}/files/${lutFilename}`;
    
    // Update Split Slider Players
    playerSliderBefore.src = beforeVideoUrl;
    playerSliderAfter.src = afterVideoUrl;
    playerSliderBefore.currentTime = 0;
    playerSliderAfter.currentTime = 0;
    sliderTimeline.value = 0;
    
    // Reset play/pause icon
    iconPlay.style.display = "block";
    iconPause.style.display = "none";
    
    // Update Side-by-Side Players
    const playerBefore = document.getElementById("player-before");
    const playerAfter = document.getElementById("player-after");
    if (playerBefore && playerAfter) {
        playerBefore.src = beforeVideoUrl;
        playerAfter.src = afterVideoUrl;
    }
    
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
    
    // Download Buttons
    const btnVid = document.getElementById("btn-download-video");
    btnVid.href = afterVideoUrl;
    btnVid.download = `${res.target_shot_id}_graded.mp4`;
    btnVid.innerText = `Download ${res.target_shot_id} Video (.mp4)`;
    
    const btnLut = document.getElementById("btn-download-lut");
    btnLut.href = lutUrl;
    btnLut.download = `${res.target_shot_id}_grade.cube`;
    btnLut.innerText = `Download ${res.target_shot_id} 3D LUT (.cube)`;
}