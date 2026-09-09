# 🎨 AutoGrader — Autonomous Multimodal Cinema Colorist

> **Autonomous Multimodal Colorist for Film & Video Production**  
> *Google Cloud "Agentic Cinema: The Blockbuster Hackathon" — Parallel (Search & Web Intelligence Track)*

🌐 **Live Web App (Google Cloud Run)**: **[https://autograder-32655487684.us-central1.run.app](https://autograder-32655487684.us-central1.run.app)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run%20Live-4285F4?logo=googlecloud&logoColor=white)](https://autograder-32655487684.us-central1.run.app)
[![Gemini](https://img.shields.io/badge/Gemini-Multimodal%20Vision-8E75B2?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Parallel](https://img.shields.io/badge/Parallel-Web%20Intelligence-F59E0B)](https://parallel.ai)
[![OpenCV](https://img.shields.io/badge/OpenCV-CIELAB%20Color%20Science-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-3D%20LUT%20Rendering-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org)

---

## 📽️ Executive Overview

In filmmaking, color grading is the bridge between technical consistency and emotional storytelling. However, indie creators and editors face two major bottlenecks:
1. **Multi-Shot Inconsistency**: Different takes or camera angles suffer from exposure drift, lighting shifts, and white balance mismatches.
2. **Translating Creative Intent into Color Science**: Converting abstract vision (*"warm desert sci-fi look with cool slate shadows and dense readable blacks"*) into precise mathematical curves and 3D LUTs requires deep color science expertise.

**AutoGrader** solves both challenges through an autonomous multimodal agent architecture. 

It is **NOT an AI video generator**. Instead, it uses **Gemini Multimodal Vision** for semantic perception, **Parallel Search** for real-time cinematography intelligence, **NumPy/OpenCV** for deterministic 32-bit floating-point CIELAB color transforms, and **FFmpeg** to render industry-standard `.cube` 3D LUTs and graded preview/delivery renders.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Browser ["User Web Interface (FastAPI + HTML5)"]
        Upload["Upload 2–4 Video Takes (MP4, MOV, MKV, WebM)"]
        Prompt["Creative Direction Prompt"]
        ProfSelect["Per-Shot Profile Dropdowns & Advisory Safety Net"]
        UI_Progress["Live Real-Time Activity Feed & Multi-Agent State Machine"]
        UI_Result["Synchronized Split-Slider Preview & Dual .cube LUT Export"]
    end

    subgraph Perception ["1. Multimodal Perception (Gemini)"]
        GeminiVision["Gemini Multimodal Vision\n• Explicit Per-Shot Scene Grouping (scene_group_id)\n• Relationship Tagging (reference, same_scene, independent_scene)\n• Context Analysis (sky/high-key vs shadow/low-key)"]
    end

    subgraph Research ["2. Web Intelligence (Parallel Search SDK)"]
        ParallelSearch["Parallel Search API\n• Real WebSearchResult.excerpts Parsing\n• Grounded Cinematography Citations\n• Honest Ungrounded Fallback"]
        GeminiSynth["Gemini Creative Synthesis\n• Sequence CreativeSpecification\n• Per-Scene Intent Modeling & Tonal Bounds\n• Shared Creative Look (Highlight/Shadow Split Tints)"]
    end

    subgraph Profiles ["3. Input Transforms & Preflight Safety"]
        InNorm["Authoritative Camera Input Transforms\n(Sony S-Log3, Apple Log / Rec.2020, DJI D-Log, DJI D-Log M, Rec.709)"]
        NormGate{"Preflight Normalization Gate\n(NORMALIZATION_VERIFIED / Non-Blocking Safety Warnings)"}
    end

    subgraph SceneAgents ["4. Dedicated Per-Scene Colorist Agents (SceneColoristAgent)"]
        MasterGrade["Grade Master Reference Shot\n(Establish Graded CIELAB Target Metrics)"]
        
        subgraph SceneDecision {"Shot Relationship?"}
            SameScene["Same-Scene Shot (is_same_scene=True)\n• Bounded CIELAB Luminance Matching (L* gain & offset)\n• Full Tonal + Chromatic Evaluation (match_colors_only=False)"]
            IndepScene["Independent Scene (is_same_scene=False)\n• Natural Exposure Preserved (L* gain=1.0, offset=0.0)\n• Match Colors & Look Only (match_colors_only=True)"]
        end

        TonalControls["Context-Aware 3-Way Tonal Controls (ThreeWayTonalParams)\n• Toe Lift (Shadows) | Midtone Gamma | Highlight Roll-off (Skies)\n• Kept Neutral by Default; Populated Only for Genuine Zonal Trims"]
    end

    subgraph RevisionStateMachine ["5. Autonomous Diagnostic Revision State Machine"]
        FastPreview["Fast Sampled-Frame Preview Render"]
        
        StateInit["INITIAL_EVALUATION"]
        FastPreview --> StateInit
        
        StateInit --> CheckAccept{"Score >= 75.0 & Health Gates Passed?"}
        CheckAccept -- "Yes" --> TermAccept["ACCEPTED"]
        CheckAccept -- "No (Rev <= 2)" --> ProposeRev["REVISION_PROPOSED\n(Diagnose Failing Component:\n• Same-Scene: ev_adj & IQR contrast\n• Independent: shadow saturation, clipping, midtone gamma)"]
        
        ProposeRev --> CheckNoOp{"Parameter Changed?"}
        CheckNoOp -- "No" --> TermNoOp["NO_ACTIONABLE_REVISION"]
        CheckNoOp -- "Yes" --> EvalProposal["Evaluate Proposed Plan"]
        
        EvalProposal --> CheckImproved{"Health-Aware Ranking:\nScore > Best Score?"}
        CheckImproved -- "Yes" --> StateImp["REVISION_IMPROVED\n(Update Best Plan)"]
        CheckImproved -- "No" --> StateRej["REVISION_REJECTED\n(Revert Proposal)"]
        
        StateImp --> LoopCheck{"Score >= 75 or Rev == 2?"}
        StateRej --> LoopCheck
        
        LoopCheck -- "Score >= 75 & Health OK" --> TermAccept
        LoopCheck -- "Rev == 2" --> TermMax["MAX_REVISIONS_REACHED\n(Render Verified Best Plan)"]
        
        TermAccept --> FinalRender["Render Delivery Video (.mp4) & Master 3D LUT (.cube)"]
        TermMax --> FinalRender
        TermNoOp --> FinalRender
    end

    Upload --> GeminiVision
    Prompt --> ParallelSearch
    ProfSelect --> InNorm
    ParallelSearch --> GeminiSynth
    
    GeminiVision --> Profiles
    InNorm --> NormGate --> MasterGrade
    GeminiSynth --> MasterGrade
    
    MasterGrade --> SceneDecision
    SameScene --> TonalControls
    IndepScene --> TonalControls
    TonalControls --> FastPreview
    FinalRender --> UI_Result
```

---

## ⚡ Key Differentiators & Autonomous Colorist Loop

1. **Dedicated Per-Scene Colorist Agents**:
   - Deploys a dedicated `SceneColoristAgent` for each distinct scene group.
   - Context-aware intelligence accepts intentional artistic choices: high-key scenes allow filmic highlight roll-off and sky clipping, while low-key scenes allow deep crushed shadow floors without triggering false health rejections.
2. **Same-Scene Bounded Luminance vs Independent Scene Color-Only Look Continuity**:
   - **Same-Scene Shots (`is_same_scene=True`)**: Deterministic bounded luminance matching ($L^*_\text{gain} \in [0.5, 2.0]$, $L^*_\text{offset} \in [-60.0, 60.0]$), full tonal + chromatic evaluation, and targeted diagnostic exposure revisions (`ev_adj`).
   - **Independent Scenes (`is_same_scene=False`)**: Preserves natural scene exposure ($L^*_\text{gain}=1.0$, $L^*_\text{offset}=0.0$), evaluating cross-scene look continuity without forcing daytime brightness onto night scenes or dark interiors.
3. **Context-Aware 3-Way Zonal Controls (`three_way`)**:
   - Explicit colorist controls for Shadows (Toe Lift, Saturation, Tint Offset), Midtones (Gamma, Contrast, Saturation, Tint Offset), and Highlights (Gain, Roll-off, Saturation, Tint Offset).
   - **Zero Double-Application**: Defaulted to neutral (`ThreeWayTonalParams()`) so scene trims and creative looks are never compounded twice; UI accurately surfaces composed totals and zonal adjustments.
4. **Authoritative Camera Input Transforms**:
   - **Sony S-Log3 / S-Gamut3.Cine**: Authoritative Sony Technical Summary inverse EOTF and chromatic adaptation matrix `MAT_SGAMUT3CINE_TO_BT709`.
   - **Apple Log / Rec.2020 (`apple_log_rec2020`)**: Truthfully reflects Apple's iPhone 15/16 Pro specification with ITU-R BT.2020 color gamut mapping via `MAT_BT2020_TO_BT709`.
   - **DJI D-Log / D-Gamut (`dji_dlog_dgamut`)**: Authoritative DJI White Paper transfer curve inverse EOTF and D-Gamut primary matrix `MAT_DGAMUT_TO_BT709`.
   - **DJI D-Log M (`dji_dlog_m`)**: Authoritative DJI D-Log M transfer curve and gamut mapping for Osmo and consumer drone systems.
   - **Generic Log & Rec.709**: Bounded experimental curve for unprofiled log and display-referred passthrough.
5. **Conservative Log Detector & Advisory Safety Net**:
   - Metadata (`ffprobe` transfer and primaries) is inspected first.
   - If metadata is inconclusive, a conservative histogram detector evaluates the conjunction of $p5 > 38.0$, chroma $< 12.0$, IQR $< 55.0$, and $p95 < 240.0$. Advisory only; never guesses camera hardware from pixel statistics alone.
   - Unresolved `auto_ask` profiles return HTTP 409 Conflict at the API boundary, guaranteeing unresolved profiles never reach rendering.
6. **Preflight Normalization Gate with Delivery Guarantee**:
   - Validates display tone distributions before paid LLM calls. Non-blocking delivery guarantee logs warnings without halting pipeline execution.
7. **Authentic Parallel Web Intelligence**: Real `WebSearchResult.excerpts` evidence is extracted and passed into the creative synthesis prompt. If Parallel is unavailable, the system reports an honest ungrounded state with zero fabricated citations.
8. **Health-Aware Autonomous Revision Ranking**:
   - Evaluates hard gates, clipping health, shadow saturation health, and look continuity.
   - Reverts ineffective revisions (`REVISION_REJECTED`) and renders verified best plans.
9. **Dual 3D LUT Exports & Genuine 1080p Demo Footage**:
   - Exports timeline-wide **Shared Creative-Look 3D LUT** (`shared_creative_look.cube`) and per-shot **Master Grade LUTs** (`shot_X_grade.cube`) in 32-bit floating-point precision.
   - Ships with genuine 1080p DJI aerial and ground footage for instant 1-click sequence demonstration.

---

## 📊 Benchmark Test Results

The suite includes a deterministic, offline benchmark runner (`tests/benchmark.py`) that exercises the entire pipeline end-to-end against local offline fixtures with zero external network dependencies.

**Benchmark Status**: `PASS` | **Sequence Average Consistency**: **97.4 / 100** | **Total Execution Time**: ~2.3s

| Shot ID | Scene Context & Relationship | Input Profile | State | Revisions | Overall Score | Tonal Sim | Chromatic Sim | Clipping Health |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **shot_A** | Daylight Reference (group_1) | `rec709` | `ACCEPTED` | 0 | **100.0 / 100** | 100.0 | 100.0 | 100.0 |
| **shot_B** | Daylight Underexposed Match (group_1) | `rec709` | `ACCEPTED` | 0 | **92.2 / 100** | 92.1 | 96.9 | 100.0 |
| **shot_C** | Golden Hour Independent Scene (group_2) | `rec709` | `ACCEPTED` | 0 | **100.0 / 100** | 100.0 | 100.0 | 100.0 |

### 1. Same-Scene Matching Mode (`same_scene_match`)
Evaluated against graded master reference target metrics:

| Scenario | Tonal Match | Chromatic Match ($\Delta E_{ab}$) | Distribution Match | Clipping Health | Overall Score | Revisions | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Identical Reference Shot** | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 | **100.0 / 100** | 0 | `ACCEPTED` |
| **Underexposed Take (-1.5 EV)** | 88.4 / 100 | 92.1 / 100 | 91.5 / 100 | 98.2 / 100 | **91.8 / 100** | 1 | `ACCEPTED` (Improved) |
| **Warm Tungsten Cast** | 86.2 / 100 | 89.7 / 100 | 90.1 / 100 | 99.0 / 100 | **89.6 / 100** | 1 | `ACCEPTED` (Improved) |

### 2. Independent Scene Look-Continuity Mode (`cross_scene_look_continuity`)
Evaluated via standardized synthetic transform probes and scene image health:

| Scenario | Shadow Split Adherence | Highlight Split Adherence | Contrast Slope Adherence | Saturation Scaling | Image Health | Overall Look Continuity | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Night Scene (Preserved Depth)** | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 | 96.5 / 100 | **99.5 / 100** | `ACCEPTED` |
| **Golden Hour Scene** | 98.2 / 100 | 99.1 / 100 | 96.8 / 100 | 95.4 / 100 | 98.0 / 100 | **97.6 / 100** | `ACCEPTED` |
| **Divergent Cyan Look (Negative Test)**| 38.4 / 100 | 40.6 / 100 | 85.2 / 100 | 82.1 / 100 | 95.0 / 100 | **58.2 / 100** | Correctly Detected Mismatch |

---

## 🛠️ Quickstart

### Prerequisites
* Python 3.11+
* FFmpeg (`ffmpeg` and `ffprobe` in system PATH)

### Installation
```bash
git clone https://github.com/Varat-S/AutoGrader.git
cd AutoGrader
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the root directory:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
PARALLEL_API_KEY=your_parallel_api_key_here
```

### Run Tests
```bash
# Run deterministic CI suite (no API keys required)
pytest -v

# Run opt-in live API smoke tests (requires GEMINI_API_KEY)
pytest -v -m live
```

### Run Web Server
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 📄 License

This project is open-source software licensed under the **[MIT License](LICENSE)**. See the [LICENSE](LICENSE) file for full details.