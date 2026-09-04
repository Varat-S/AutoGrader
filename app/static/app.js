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

// Advisory Warning Banner (P0-C)
const bannerWarning = document.getElementById("profile-warning-banner");
const bannerTitle = document.getElementById("banner-title");
const bannerMessage = document.getElementById("banner-message");
const btnBannerFix = document.getElementById("btn-banner-fix");
const btnBannerDismiss = document.getElementById("btn-banner-dismiss");
let currentAssessments = [];

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
            const clips = data.all_clips || data.loaded || [];
            updateClipsList(clips);
            btnRun.disabled = (clips.length === 0);
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

    // Banner Actions (P0-C)
    if (btnBannerFix) {
        btnBannerFix.addEventListener("click", () => {
            currentAssessments.forEach((ass, idx) => {
                const sel = document.querySelector(`.clip-profile-select[data-shot-index="${idx}"]`);
                if (sel && ass.selected_profile) {
                    sel.value = ass.selected_profile;
                    sel.dataset.userModified = "true";
                }
            });
            if (bannerWarning) bannerWarning.style.display = "none";
        });
    }

    if (btnBannerDismiss) {
        btnBannerDismiss.addEventListener("click", () => {
            if (bannerWarning) bannerWarning.style.display = "none";
        });
    }

    // Run Workflow
    btnRun.addEventListener("click", async () => {
        if (!currentJobId || btnRun.disabled) return;
        btnRun.disabled = true;
        
        const refVal = refSelect.value === "auto" ? null : parseInt(refSelect.value);
        const colorProfileVal = colorSelect.value;
        
        const inputProfiles = [];
        const shotSelects = document.querySelectorAll(".clip-profile-select");
        shotSelects.forEach((sel, idx) => {
            inputProfiles.push({ shot_index: idx, profile: sel.value });
        });
        
        try {
            const res = await fetch(`/api/jobs/${currentJobId}/run`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    creative_prompt: promptInput.value,
                    reference_index: refVal,
                    color_profile: colorProfileVal,
                    input_profiles: inputProfiles
                })
            });
            
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                alert(errData.detail || "Error starting grading job");
                btnRun.disabled = false;
                return;
            }
            
            document.getElementById("agent-activity-panel").style.display = "block";
            document.getElementById("results-panel").style.display = "none";
            startPolling(currentJobId);
        } catch (e) {
            console.error("Run error:", e);
            btnRun.disabled = false;
        }
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
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            alert(errData.detail || "Failed to upload video clip(s).");
            return;
        }
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
        if (bannerWarning) bannerWarning.style.display = "none";
        return;
    }
    
    filenames.forEach((fname, idx) => {
        const shotId = `shot_${String.fromCharCode(65 + idx)}`;
        const row = document.createElement("div");
        row.className = "clip-row";
        
        const rowLeft = document.createElement("div");
        rowLeft.className = "clip-row-left";
        
        const strong = document.createElement("strong");
        strong.textContent = shotId;
        rowLeft.appendChild(strong);
        
        const nameSpan = document.createElement("span");
        nameSpan.className = "clip-row-name";
        nameSpan.textContent = fname;
        nameSpan.title = fname;
        rowLeft.appendChild(nameSpan);
        
        const rowActions = document.createElement("div");
        rowActions.className = "clip-row-actions";
        
        const tag = document.createElement("span");
        tag.className = "clip-tag";
        tag.textContent = fname.split(".").pop().toUpperCase();
        rowActions.appendChild(tag);
        
        const sel = document.createElement("select");
        sel.className = "clip-profile-select";
        sel.dataset.shotIndex = idx;
        
        const profileOptions = [
            { value: "auto_ask", label: "Auto-detect" },
            { value: "rec709", label: "Rec.709 / Display" },
            { value: "sony_slog3_sgamut3cine", label: "Sony S-Log3" },
            { value: "apple_log_apple_wide_gamut", label: "Apple Log" },
            { value: "generic_log_experimental", label: "Generic Log / Flat" }
        ];
        
        profileOptions.forEach(optData => {
            const opt = document.createElement("option");
            opt.value = optData.value;
            opt.textContent = optData.label;
            sel.appendChild(opt);
        });
        
        sel.addEventListener("change", () => {
            sel.dataset.userModified = "true";
        });
        
        rowActions.appendChild(sel);
        
        row.appendChild(rowLeft);
        row.appendChild(rowActions);
        clipsList.appendChild(row);
        
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `Shot ${String.fromCharCode(65 + idx)} (${fname})`;
        refSelect.appendChild(opt);
    });
    
    btnRun.disabled = false;
    
    // Asynchronously assess input profiles and display safety warnings
    fetchProfileAssessments();
}

async function fetchProfileAssessments() {
    if (!currentJobId) return;
    try {
        const res = await fetch(`/api/jobs/${currentJobId}/assess_profiles`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "success" && Array.isArray(data.assessments)) {
            currentAssessments = data.assessments;
            let hasWarning = false;
            let warningText = "";
            
            data.assessments.forEach((ass, idx) => {
                const sel = document.querySelector(`.clip-profile-select[data-shot-index="${idx}"]`);
                if (sel && !sel.dataset.userModified) {
                    if (ass.selected_profile && ass.selected_profile !== "rec709") {
                        sel.value = ass.selected_profile;
                    }
                }
                if (ass.profile_mismatch_warning) {
                    hasWarning = true;
                    warningText += `${ass.shot_id}: ${ass.warning_message} `;
                }
            });
            
            if (hasWarning && bannerWarning) {
                bannerTitle.textContent = "Input Profile Advisory Warning";
                bannerMessage.textContent = warningText || "Log-encoded or flat footage detected under display profile.";
                bannerWarning.style.display = "flex";
            } else if (bannerWarning) {
                bannerWarning.style.display = "none";
            }
        }
    } catch (e) {
        console.warn("Could not fetch profile assessments:", e);
    }
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
    const statusBadge = document.getElementById("agent-status-badge");
    statusBadge.textContent = "Completed";
    statusBadge.className = "status-badge badge-done";
    
    // Creative Spec Card (Safe DOM Construction)
    const spec = result.creative_specification;
    const specCard = document.getElementById("creative-spec-card");
    specCard.innerHTML = "";
    
    const titleRow = document.createElement("div");
    titleRow.className = "spec-title-row";
    const titleDiv = document.createElement("div");
    titleDiv.className = "spec-title";
    titleDiv.textContent = `Look: "${spec.look_title}" (Reference Shot: ${result.reference_shot_id})`;
    titleRow.appendChild(titleDiv);
    specCard.appendChild(titleRow);
    
    const descP = document.createElement("p");
    descP.style.fontSize = "0.85rem";
    descP.style.color = "#64748b";
    descP.style.marginBottom = "0.5rem";
    descP.textContent = spec.target_aesthetic;
    specCard.appendChild(descP);
    
    const badgesDiv = document.createElement("div");
    badgesDiv.className = "spec-badges";
    
    const badgeItems = [
        `Contrast: ${spec.contrast_intent}x`,
        `Saturation: ${spec.saturation_intent}x`,
        `Highlights: ${spec.highlight_bias}`,
        `Shadows: ${spec.shadow_bias}`
    ];
    
    badgeItems.forEach(bText => {
        const bSpan = document.createElement("span");
        bSpan.className = "spec-badge";
        bSpan.textContent = bText;
        badgesDiv.appendChild(bSpan);
    });
    specCard.appendChild(badgesDiv);
    
    // Citations (Safe URL validation)
    const citationsList = document.getElementById("citations-list");
    citationsList.innerHTML = "";
    if (result.research_citations && result.research_citations.length > 0) {
        result.research_citations.forEach(c => {
            const link = document.createElement("a");
            link.className = "citation-link";
            if (c.url && (c.url.startsWith("http://") || c.url.startsWith("https://"))) {
                link.href = c.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
            } else {
                link.href = "#";
            }
            link.textContent = `Source: ${c.title}`;
            citationsList.appendChild(link);
        });
    } else {
        const fallbackText = document.createElement("div");
        fallbackText.style.fontSize = "0.85rem";
        fallbackText.style.color = "#94a3b8";
        fallbackText.textContent = "Parallel grounding unavailable — synthesized with ungrounded color science principles.";
        citationsList.appendChild(fallbackText);
    }
    
    // Shared Look LUT button
    const btnSharedLut = document.getElementById("btn-download-shared-lut");
    if (btnSharedLut && result.shared_lut_path) {
        const sharedFilename = result.shared_lut_path.split("/").pop().split("\\").pop();
        btnSharedLut.href = `/api/jobs/${currentJobId}/files/${sharedFilename}`;
        btnSharedLut.download = `shared_creative_look.cube`;
    }
    
    // Shot Tabs
    const shotTabs = document.getElementById("shot-tabs");
    shotTabs.innerHTML = "";
    result.results.forEach((r, idx) => {
        const isMaster = (r.target_shot_id === result.reference_shot_id);
        const btn = document.createElement("button");
        btn.className = `shot-tab ${idx === 0 ? "active" : ""}`;
        btn.textContent = `${r.target_shot_id}${isMaster ? " (Master Ref)" : ""}`;
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
    
    document.getElementById("label-source-shot").textContent = `Source: ${res.target_shot_id} (${sourceFilename})`;
    document.getElementById("label-graded-shot").textContent = `Graded: ${res.target_shot_id}`;
    
    // Scores
    const beforeScore = Math.round(res.before_consistency.overall_score);
    const afterScore = Math.round(res.after_consistency.overall_score);
    document.getElementById("score-before").textContent = beforeScore;
    document.getElementById("score-after").textContent = afterScore;
    
    const delta = afterScore - beforeScore;
    const stateText = res.state ? ` (${res.state})` : "";
    if (res.revisions_performed && res.revisions_performed > 0) {
        document.getElementById("score-delta").textContent = `${res.state || "REVISION"}: Harmonized over ${res.revisions_performed} revision passes`;
    } else {
        document.getElementById("score-delta").textContent = delta >= 0 ? `+${delta} points consistency${stateText}` : `Master Style Established${stateText}`;
    }
    
    document.getElementById("metric-tone").textContent = `${Math.round(res.after_consistency.tonal_similarity)} / 100`;
    document.getElementById("metric-chroma").textContent = `${Math.round(res.after_consistency.chromatic_similarity)} / 100`;
    document.getElementById("metric-health").textContent = `${Math.round(res.after_consistency.clipping_health)} / 100`;
    
    document.getElementById("explanation-text").textContent = res.explanation;
    
    // Download Buttons
    const btnVid = document.getElementById("btn-download-video");
    btnVid.href = afterVideoUrl;
    btnVid.download = `${res.target_shot_id}_graded.mp4`;
    btnVid.textContent = `Download ${res.target_shot_id} Video (.mp4)`;
    
    const btnLut = document.getElementById("btn-download-lut");
    btnLut.href = lutUrl;
    btnLut.download = `${res.target_shot_id}_grade.cube`;
    btnLut.textContent = `Download ${res.target_shot_id} 3D LUT (.cube)`;
}