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
            promptChips.forEach(c => c.classList.remove("chip-active"));
            chip.classList.add("chip-active");
            promptInput.value = chip.getAttribute("data-prompt");
            promptInput.focus();
        });
    });

    promptInput.addEventListener("input", () => {
        const val = promptInput.value.trim();
        promptChips.forEach(c => {
            if (c.getAttribute("data-prompt").trim() === val) {
                c.classList.add("chip-active");
            } else {
                c.classList.remove("chip-active");
            }
        });
    });

    // Sequence default profile propagation to unmodified shots
    colorSelect.addEventListener("change", () => {
        const val = colorSelect.value;
        const shotSelects = document.querySelectorAll(".clip-profile-select");
        shotSelects.forEach(sel => {
            if (!sel.dataset.userModified) {
                if (val === "auto") {
                    sel.value = "auto_ask";
                } else {
                    sel.value = val;
                }
            }
        });
    });

    // Apply Recommended Profiles Actions
    const applyRecommendedProfiles = () => {
        if (!currentAssessments || currentAssessments.length === 0) return;
        currentAssessments.forEach((ass, idx) => {
            const sel = document.querySelector(`.clip-profile-select[data-shot-index="${idx}"]`);
            if (sel) {
                const rec = ass.recommended_profile || ass.metadata_recommendation || (ass.selected_profile !== "auto_ask" ? ass.selected_profile : "rec709");
                sel.value = rec;
                sel.dataset.userModified = "true";
            }
        });
        if (bannerWarning) bannerWarning.style.display = "none";
    };

    if (btnBannerFix) {
        btnBannerFix.addEventListener("click", applyRecommendedProfiles);
    }
    const btnApplyRec = document.getElementById("btn-apply-recommended");
    if (btnApplyRec) {
        btnApplyRec.addEventListener("click", applyRecommendedProfiles);
    }

    if (btnBannerDismiss) {
        btnBannerDismiss.addEventListener("click", () => {
            if (bannerWarning) bannerWarning.style.display = "none";
        });
    }

    // Run Workflow
    btnRun.addEventListener("click", async () => {
        if (!currentJobId || btnRun.disabled) return;
        
        const shotSelects = document.querySelectorAll(".clip-profile-select");
        const unresolvedShots = [];
        shotSelects.forEach((sel, idx) => {
            if (sel.value === "auto_ask") {
                unresolvedShots.push(`Shot ${String.fromCharCode(65 + idx)}`);
            }
        });
        
        if (unresolvedShots.length > 0) {
            alert(`Please select or confirm camera profiles for all shots before grading.\n\nUnresolved shots: ${unresolvedShots.join(", ")}`);
            return;
        }

        btnRun.disabled = true;
        
        const refVal = refSelect.value === "auto" ? null : parseInt(refSelect.value);
        const colorProfileVal = colorSelect.value;
        
        const inputProfiles = [];
        shotSelects.forEach((sel, idx) => {
            inputProfiles.push({ shot_index: idx, profile: sel.value, user_confirmed: true });
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
            { value: "auto_ask", label: "Auto-detect / Ask if unsure" },
            { value: "rec709", label: "Rec.709 / Display" },
            { value: "sony_slog3_sgamut3cine", label: "Sony S-Log3 / S-Gamut3.Cine" },
            { value: "apple_log_rec2020", label: "Apple Log / Rec.2020" },
            { value: "dji_dlog_dgamut", label: "DJI D-Log / D-Gamut" },
            { value: "dji_dlog_m_rec709", label: "DJI D-Log M / Rec.709" },
            { value: "generic_log_experimental", label: "Generic Log / Flat" }
        ];
        
        profileOptions.forEach(optData => {
            const opt = document.createElement("option");
            opt.value = optData.value;
            opt.textContent = optData.label;
            sel.appendChild(opt);
        });
        
        const seqVal = colorSelect.value;
        sel.value = seqVal === "auto" ? "auto_ask" : seqVal;
        
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
                    if (ass.resolved_profile) {
                        sel.value = ass.resolved_profile;
                    } else if (ass.recommended_profile) {
                        sel.value = ass.recommended_profile;
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
    titleDiv.textContent = `Shared Sequence Look: "${spec.look_title}" (Master Reference: ${result.reference_shot_id})`;
    titleRow.appendChild(titleDiv);
    specCard.appendChild(titleRow);
    
    const descP = document.createElement("p");
    descP.style.fontSize = "0.85rem";
    descP.style.color = "#64748b";
    descP.style.marginBottom = "0.5rem";
    descP.textContent = `${spec.target_aesthetic} — Sequence-wide creative transform is immutable once approved.`;
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
            const item = document.createElement("div");
            item.className = "citation-item";
            item.style.marginBottom = "0.5rem";

            const link = document.createElement("a");
            link.className = "citation-link";
            const validUrl = c.url && (c.url.startsWith("http://") || c.url.startsWith("https://"));
            if (validUrl) {
                link.href = c.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
            } else {
                link.href = "#";
            }
            link.textContent = `Source: ${c.title || "Cinematography Reference"}`;
            item.appendChild(link);

            if (c.extracted_principle) {
                const princ = document.createElement("div");
                princ.style.fontSize = "0.78rem";
                princ.style.color = "#94a3b8";
                princ.style.marginTop = "0.15rem";
                princ.textContent = c.extracted_principle;
                item.appendChild(princ);
            }
            if (c.influence) {
                const infl = document.createElement("div");
                infl.style.fontSize = "0.74rem";
                infl.style.color = "#38bdf8";
                infl.style.marginTop = "0.1rem";
                infl.textContent = `Influence: ${c.influence}`;
                item.appendChild(infl);
            }
            citationsList.appendChild(item);
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

    // Prefer matched browser proxies for synchronized split slider wipe
    let sliderBeforeUrl = beforeVideoUrl;
    let sliderAfterUrl = afterVideoUrl;
    if (res.before_proxy_path) {
        const bName = res.before_proxy_path.split("/").pop().split("\\").pop();
        sliderBeforeUrl = `/api/jobs/${jobId}/files/${bName}`;
    }
    if (res.after_proxy_path) {
        const aName = res.after_proxy_path.split("/").pop().split("\\").pop();
        sliderAfterUrl = `/api/jobs/${jobId}/files/${aName}`;
    }
    
    // Update Split Slider Players
    playerSliderBefore.src = sliderBeforeUrl;
    playerSliderAfter.src = sliderAfterUrl;
    playerSliderBefore.currentTime = 0;
    playerSliderAfter.currentTime = 0;
    sliderTimeline.value = 0;
    
    // Reset play/pause icon
    iconPlay.style.display = "block";
    iconPause.style.display = "none";
    
    // Update Side-by-Side Players with matched browser proxies
    const playerBefore = document.getElementById("player-before");
    const playerAfter = document.getElementById("player-after");
    if (playerBefore && playerAfter) {
        playerBefore.src = sliderBeforeUrl;
        playerAfter.src = sliderAfterUrl;
    }
    
    document.getElementById("label-source-shot").textContent = `Ungraded Proxy: ${res.target_shot_id}`;
    document.getElementById("label-graded-shot").textContent = `Graded Proxy: ${res.target_shot_id}`;
    
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
    
    // Mode-Appropriate Metrics
    const isSameScene = (res.evaluation_mode === "same_scene_match");
    if (isSameScene) {
        document.querySelector("#metric-tone").previousElementSibling.textContent = "Tonal & Exposure Similarity";
        document.querySelector("#metric-tone").nextElementSibling.textContent = "Quantile-quantile L* alignment";
        document.getElementById("metric-tone").textContent = `${Math.round(res.after_consistency.tonal_similarity)} / 100`;

        document.querySelector("#metric-chroma").previousElementSibling.textContent = "Chromatic CIELAB Harmony";
        document.querySelector("#metric-chroma").nextElementSibling.textContent = "Delta E centroid convergence";
        document.getElementById("metric-chroma").textContent = `${Math.round(res.after_consistency.chromatic_similarity)} / 100`;

        document.querySelector("#metric-health").previousElementSibling.textContent = "Clipping & Image Health";
        document.querySelector("#metric-health").nextElementSibling.textContent = "Protected shadows & highlights";
        document.getElementById("metric-health").textContent = `${Math.round(res.after_consistency.clipping_health)} / 100`;
    } else {
        document.querySelector("#metric-tone").previousElementSibling.textContent = "Creative Contrast Adherence";
        document.querySelector("#metric-tone").nextElementSibling.textContent = "Probe tone response invariant";
        const toneVal = res.look_continuity ? Math.round(res.look_continuity.contrast_slope_adherence) : Math.round(res.after_consistency.tonal_similarity);
        document.getElementById("metric-tone").textContent = `${toneVal} / 100`;

        document.querySelector("#metric-chroma").previousElementSibling.textContent = "Probe Chromatic Harmony";
        document.querySelector("#metric-chroma").nextElementSibling.textContent = "Split-tone & saturation tracking";
        const chromaVal = res.look_continuity ? Math.round(res.look_continuity.probe_chromatic_harmony) : Math.round(res.after_consistency.chromatic_similarity);
        document.getElementById("metric-chroma").textContent = `${chromaVal} / 100`;

        document.querySelector("#metric-health").previousElementSibling.textContent = "Scene Image Health";
        document.querySelector("#metric-health").nextElementSibling.textContent = res.scene_health && res.scene_health.hard_gates_passed ? "Hard gates passed (exposure, midtones, shadow sat)" : "Gate failure detected";
        const healthVal = res.scene_health ? Math.round(res.scene_health.overall_score) : Math.round(res.after_consistency.clipping_health);
        document.getElementById("metric-health").textContent = `${healthVal} / 100`;
    }
    
    // Effective Grade Summary Panel (Safe DOM construction, no innerHTML)
    const effPanel = document.getElementById("effective-grade-panel");
    if (effPanel && res.grade_summary) {
        effPanel.style.display = "block";
        effPanel.innerHTML = "";
        const summ = res.grade_summary;

        const headerDiv = document.createElement("div");
        headerDiv.style.display = "flex";
        headerDiv.style.justifyContent = "space-between";
        headerDiv.style.alignItems = "center";
        headerDiv.style.marginBottom = "0.6rem";

        const titleDiv = document.createElement("div");
        titleDiv.style.fontWeight = "700";
        titleDiv.style.fontSize = "0.95rem";
        titleDiv.style.color = "var(--text-primary)";
        titleDiv.textContent = `Effective Grade Breakdown for ${res.target_shot_id}`;
        headerDiv.appendChild(titleDiv);

        const badgeSpan = document.createElement("span");
        badgeSpan.className = "spec-badge";
        badgeSpan.style.background = "#e0f2fe";
        badgeSpan.style.color = "#0369a1";
        badgeSpan.style.fontWeight = "600";
        badgeSpan.textContent = `Profile: ${summ.camera_profile}`;
        headerDiv.appendChild(badgeSpan);
        effPanel.appendChild(headerDiv);

        const gridDiv = document.createElement("div");
        gridDiv.className = "effective-grid";

        // Cell 1: Technical Balance
        const cell1 = document.createElement("div");
        cell1.className = "effective-cell";
        const c1Title = document.createElement("div");
        c1Title.className = "effective-cell-title";
        c1Title.textContent = "1. Technical Balance";
        const c1Val = document.createElement("div");
        c1Val.className = "effective-cell-val";
        const expSign1 = summ.technical_balance.exposure_ev > 0 ? "+" : "";
        const tSign = summ.technical_balance.temperature > 0 ? "+" : "";
        const tintSign = summ.technical_balance.tint > 0 ? "+" : "";
        const l1 = document.createElement("div");
        l1.textContent = `Exp: ${expSign1}${summ.technical_balance.exposure_ev.toFixed(2)} EV`;
        const l2 = document.createElement("div");
        l2.textContent = `WB: ${tSign}${summ.technical_balance.temperature.toFixed(1)} / ${tintSign}${summ.technical_balance.tint.toFixed(1)}`;
        c1Val.appendChild(l1);
        c1Val.appendChild(l2);
        cell1.appendChild(c1Title);
        cell1.appendChild(c1Val);
        gridDiv.appendChild(cell1);

        // Cell 2: Shared Creative Look (Immutable)
        const cell2 = document.createElement("div");
        cell2.className = "effective-cell";
        const c2Title = document.createElement("div");
        c2Title.className = "effective-cell-title";
        c2Title.textContent = "2. Shared Creative Look (Immutable)";
        const c2Val = document.createElement("div");
        c2Val.className = "effective-cell-val";
        const l3 = document.createElement("div");
        l3.textContent = `Contrast: ${summ.shared_creative_look.contrast.toFixed(2)}x`;
        const l4 = document.createElement("div");
        l4.textContent = `Sat: ${summ.shared_creative_look.saturation.toFixed(2)}x`;
        const l5 = document.createElement("div");
        l5.textContent = `Splits: ${summ.shared_creative_look.highlight_bias} / ${summ.shared_creative_look.shadow_bias}`;
        c2Val.appendChild(l3);
        c2Val.appendChild(l4);
        c2Val.appendChild(l5);
        cell2.appendChild(c2Title);
        cell2.appendChild(c2Val);
        gridDiv.appendChild(cell2);

        // Cell 3: Scene Trim
        const cell3 = document.createElement("div");
        cell3.className = "effective-cell";
        const c3Title = document.createElement("div");
        c3Title.className = "effective-cell-title";
        c3Title.textContent = "3. Scene Trim";
        const c3Val = document.createElement("div");
        c3Val.className = "effective-cell-val";
        const expSign3 = summ.scene_trim.trim_exposure_ev > 0 ? "+" : "";
        const l6 = document.createElement("div");
        l6.textContent = `Exp: ${expSign3}${summ.scene_trim.trim_exposure_ev.toFixed(2)} EV`;
        const l7 = document.createElement("div");
        l7.textContent = `Contrast: ${summ.scene_trim.trim_contrast.toFixed(2)}x`;
        const l8 = document.createElement("div");
        l8.textContent = `Sat: ${summ.scene_trim.trim_saturation.toFixed(2)}x`;
        c3Val.appendChild(l6);
        c3Val.appendChild(l7);
        c3Val.appendChild(l8);
        cell3.appendChild(c3Title);
        cell3.appendChild(c3Val);
        gridDiv.appendChild(cell3);

        // Cell 4: Composed Effective Grade
        const cell4 = document.createElement("div");
        cell4.className = "effective-cell";
        cell4.style.borderLeft = "3px solid var(--accent-emerald)";
        const c4Title = document.createElement("div");
        c4Title.className = "effective-cell-title";
        c4Title.style.color = "var(--accent-emerald)";
        c4Title.textContent = "4. Composed Effective Grade";
        const c4Val = document.createElement("div");
        c4Val.className = "effective-cell-val";
        c4Val.style.fontWeight = "600";
        const expSign4 = summ.effective_exposure_ev > 0 ? "+" : "";
        const l9 = document.createElement("div");
        l9.textContent = `Net Exp: ${expSign4}${summ.effective_exposure_ev.toFixed(2)} EV`;
        const l10 = document.createElement("div");
        l10.textContent = `Net Contrast: ${summ.effective_contrast.toFixed(2)}x`;
        const l11 = document.createElement("div");
        l11.textContent = `Net Saturation: ${summ.effective_saturation.toFixed(2)}x`;
        c4Val.appendChild(l9);
        c4Val.appendChild(l10);
        c4Val.appendChild(l11);
        cell4.appendChild(c4Title);
        cell4.appendChild(c4Val);
        gridDiv.appendChild(cell4);

        effPanel.appendChild(gridDiv);
    } else if (effPanel) {
        effPanel.style.display = "none";
    }
    
    document.getElementById("explanation-text").textContent = res.explanation;
    
    // Download Buttons
    const btnVid = document.getElementById("btn-download-video");
    btnVid.href = afterVideoUrl;
    btnVid.download = `${res.target_shot_id}_graded.mp4`;
    btnVid.textContent = `Download Delivery Video (.mp4)`;
    
    const btnLut = document.getElementById("btn-download-lut");
    btnLut.href = lutUrl;
    btnLut.download = `${res.target_shot_id}_grade.cube`;
    btnLut.textContent = `Download ${res.target_shot_id} 3D LUT (.cube)`;
}