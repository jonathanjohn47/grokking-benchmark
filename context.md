# Session Summary — Thesis Gantt Chart & Setup

## Identity & Context

- **Name:** Jonathan John
- **Programme:** M.Sc. Artificial Intelligence, IU Internationale Hochschule (2nd thesis attempt)
- **Thesis Title:** *A Unified Benchmark of Grokking Predictors in Neural Networks*
- **Subtitle:** A head-to-head empirical comparison of 9 published grokking predictors under a unified benchmark protocol
- **Supervisor:** Prof. Dr.-Ing. Sheikh Faisal Rashid (AI, Berlin campus)
- **Official Start Date:** June 22, 2026
- **Location:** Jammu, India

---

## Supervisor Instructions (from email thread)

- Use the existing **4-predictor baseline** as a starting point, then extend to **9 predictors**
- Prepare a complete **thesis timeline (Gantt chart)** with deliverables and milestones
- Two open questions from Prof. Rashid:
    1. What was the **previous thesis topic**? (2nd topic must be different)
    2. Have I **moved to Jammu**? (noted in signature)

---

## What Was Completed This Session

### 1. Experiment Start Plan
- Full phase-by-phase breakdown (Phases 0–7) with immediate next actions
- **Critical gate identified:** reproduce Nanda et al. grokking on `(a+b) mod 97` before any predictor work begins
- Predictor implementation order defined (easy → hard): L2 Norm → Dropout → Spectral → AGE → HTSR Alpha → Correlation Traps → Weight-PCA → Higher-MI → Commutator Defect

### 2. Thesis Gantt Chart
- Built as a **Google Apps Script** (`.gs`) that runs inside Google Sheets
- Produces a fully color-coded Gantt with 6 phases, 20 tasks, milestone markers, legend, and notes bar
- Hosted in Google Drive:
  `https://docs.google.com/spreadsheets/d/11Pst2P18QE3N7lbhnqwdkUN6OONTyTSPNCkSjTz7-RI`
- To rebuild: open sheet → **Extensions → Apps Script** → paste script → **Run → `buildGantt`**

### 3. Preferences Confirmed
- Always use **Google Sheets** (never Excel/xlsx) for spreadsheet tasks

---

## Gantt Chart Structure

| Phase | Color | Period | Key Deliverable |
|---|---|---|---|
| Ph 1 Setup | Blue | Jun–Jul | Working PyTorch MPS pipeline; canonical grokking reproduced |
| Ph 2 Baseline | Green | Aug | 4-predictor baseline results table |
| Ph 3 Predictors | Orange | Aug–Sep | All 9 predictors implemented + unit-tested |
| Ph 4 Sweep | Amber | Oct | ~80 training runs; Plots 1, 2, 3 |
| Ph 5 Ensemble | Purple | Oct–Nov | Meta-predictor + anti-grokking; Plot 4 |
| Ph 6 Writing | Dark Red | Oct–Nov | Thesis draft → revisions → submission |

### Milestones (red M cells)

| Milestone | When | Gate |
|---|---|---|
| M1 | Jul W4 | 🚦 Canonical grokking reproduced — do not proceed until confirmed |
| M2 | Aug W4 | Baseline 4-predictor results complete |
| M3 | Sep W4 | All 9 predictors verified |
| M4 | Oct W4 | Leaderboard + Plots 1 & 2 done |
| M5 | Nov W3 | All 4 canonical plots + ensemble complete |
| M6 | Nov W4 | **Thesis submitted ✓** |

---

## Pending (Not Yet Done)

- [ ] Reply to Prof. Rashid's two questions (previous thesis topic + Jammu clarification)
- [ ] Environment setup — Git repo, PyTorch MPS, requirements.txt
- [ ] Reproduce canonical Nanda et al. grokking (first critical gate)

---

## Tools & Preferences

| Tool | Preference |
|---|---|
| Spreadsheets | Google Sheets (never Excel) |
| Prompts | Opencode prompt format by default |
| Implementation | Only on explicit request |