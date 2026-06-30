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

## Session 2 — Phase 1 Implementation (June 30, 2026)

### Completed
- ✅ Virtual environment created (`.venv`) and all `requirements.txt` packages installed
- ✅ MPS confirmed working (`torch.backends.mps.is_available()` → `True`)
- ✅ Project structure created: `src/data/`, `src/models/`, `src/train.py`
- ✅ `generate_pairs(number)` written in `src/data/modular_arithmetic.py` using `yield`
  - Loops `i` in `range(number)`, `j` in `range(number)`
  - Yields `(i, j, (i+j) % number)` — 9409 triples for p=97

### Also Completed (Session 3 — June 30, 2026)
- ✅ `get_dataloaders(number_of_tuples)` written in `src/data/modular_arithmetic.py`
  - Generates all 9409 pairs, splits 30% train / 70% val
  - Returns `train, test` as plain lists (DataLoader wrapping still pending)
  - Jonathan understands why 30/70 split: grokking requires forcing generalization, not memorization
- ✅ `train.py` cleaned up: duplicate `generate_pairs` removed, correct import from `data.modular_arithmetic`
- ✅ Exposed API key in `train.py` comment removed
- ✅ Conceptual understanding confirmed:
  - `p=97` is prime → forms a finite field → clean algebraic structure
  - 97×97 = 9409 total input pairs (full input space)
  - `DataLoader` is a PyTorch built-in, not written by hand

### In Progress
- 🔄 `get_dataloaders` returns plain lists — needs to wrap in PyTorch `DataLoader` objects
  - Requires a `Dataset` class with `__len__` and `__getitem__` first

### Next Session — Pick Up Here
1. Write `Dataset` class in `src/data/modular_arithmetic.py`, wrap `train`/`test` in `DataLoader`
2. Write `src/models/transformer.py` (Embedding → Attn → MLP → Output Head)
3. Write training loop in `src/train.py` (setup → train loop → logging → plot grokking curve)
4. Run training and observe grokking curve (M1 gate)

### Jonathan's Coding Style Note
- Jonathan is learning Python while implementing — explain concepts before code
- He writes code himself; Claude only helps when stuck
- Prefers to understand the *purpose* of each function before writing it

---

## Session 4 — June 30, 2026 (Python Concepts)

### Covered
- ✅ What goes inside a `Dataset` class: `__init__`, `__len__`, `__getitem__` — explained with purpose before code
- ✅ How to write a Python class — `__init__`, methods, instances explained
- ✅ What `self` is — long discussion in Hinglish, final understanding:
  - `self` is needed because `bark()` function andar se nahi jaanta ki kis object ke liye chal raha hai
  - Tu bahar se `rex.bark()` likhta hai — Python `rex` ko function ke andar `self` ke roop mein bhej deta hai
  - `self` woh bridge hai between the call outside and the data inside

### Jonathan's Learning Style Notes (Updated)
- Explains in Hindi/Hinglish when confused — respond in same language
- Very literal thinker — analogies backfire, direct logical explanation works better
- Asks "why does this even exist?" before accepting syntax — answer that first
- Does not move on until genuinely understood

### Next Session — Pick Up Here (unchanged)
1. Write `ModularArithmeticDataset` class in `src/data/modular_arithmetic.py`
2. Wrap `train`/`test` in `DataLoader`
3. Write `src/models/transformer.py`
4. Write training loop in `src/train.py`
5. Run and confirm grokking curve (M1 gate)

---

## Session 5 — June 30, 2026 (ModularArithmeticDataset — Concepts)

### Covered
- ✅ What variables are needed inside `ModularArithmeticDataset.__init__`
  - Only one: `self.data = data`
  - `__len__` returns `len(self.data)`, `__getitem__` returns `self.data[index]`
- ✅ What `__init__` does — explained directly:
  - Python automatically calls `__init__` the moment an object is created
  - Its job: store starting values inside the object so other methods can use them
  - Jonathan asked "agar `__init__` naa likhu to?" → explained `AttributeError` and `TypeError` that result
- ✅ Where to write the class:
  - `src/data/modular_arithmetic.py` — same file as `generate_pairs` and `get_dataloaders`
  - No new folder needed; class belongs in data module
- ✅ Git lock file issue encountered:
  - `HEAD.lock` was blocking commits
  - Fix: `rm .git/HEAD.lock` then re-run `git commit`

### Jonathan's Learning Style Notes (Further Updated)
- Asks "agar ye naa karu to kya hoga?" — always answer the negative case directly
- Wants to understand each concept fully before writing any code
- Hinglish explanations work best when he phrases questions in Hindi

### Next Session — Pick Up Here
1. Write `ModularArithmeticDataset` class in `src/data/modular_arithmetic.py`
2. Wrap `train`/`test` in `DataLoader`
3. Write `src/models/transformer.py`
4. Write training loop in `src/train.py`
5. Run and confirm grokking curve (M1 gate)

---

## Session 6 — June 30, 2026 (ModularArithmeticDataset — Implementation)

### Completed
- ✅ `ModularArithmeticDataset` class written in a **separate file**: `src/data/dataset.py` (Jonathan chose to split it out, not keep it in `modular_arithmetic.py`)
- ✅ Two bugs fixed during creation:
  - Circular import removed (`from data.dataset import ModularArithmeticDataset` was at top of the same file — deleted)
  - Import path corrected from `src.data.modular_arithmetic` → `data.modular_arithmetic` to match project convention in `train.py`
- ✅ Final class structure in `src/data/dataset.py`:
  - `__init__(self, number_of_tuples)` — calls `generate_pairs`, stores result as `self.data`
  - `__len__` — returns `len(self.data)`
  - `__getitem__(self, idx)` — returns `self.data[idx]`

### Next Session — Pick Up Here
1. Wrap `train`/`test` lists in PyTorch `DataLoader` (import `ModularArithmeticDataset` from `data.dataset`)
2. Write `src/models/transformer.py` (Embedding → Attn → MLP → Output Head)
3. Write training loop in `src/train.py`
4. Run and confirm grokking curve (M1 gate)

---

## Pending (Not Yet Done)

- [ ] Reply to Prof. Rashid's two questions (previous thesis topic + Jammu clarification)
- [ ] Wrap `train`/`test` in PyTorch `DataLoader` objects (`ModularArithmeticDataset` now ready in `src/data/dataset.py`)
- [ ] Write transformer model and training loop
- [ ] Reproduce canonical Nanda et al. grokking (M1 gate)

---

## Tools & Preferences

| Tool | Preference |
|---|---|
| Spreadsheets | Google Sheets (never Excel) |
| Prompts | Opencode prompt format by default |
| Implementation | Only on explicit request |