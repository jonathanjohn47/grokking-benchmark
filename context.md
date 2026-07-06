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

## Project Plan (unchanged reference material)

- **Critical gate (M1):** reproduce Nanda et al. grokking on `(a+b) mod 97` before any predictor work begins
- **Predictor implementation order (easy → hard):** L2 Norm → Dropout → Spectral → AGE → HTSR Alpha → Correlation Traps → Weight-PCA → Higher-MI → Commutator Defect
- **Hardware target:** PyTorch + Apple Silicon MPS
- **Spreadsheet policy:** Always Google Sheets, never Excel/xlsx
- **Thesis Gantt chart:** Google Apps Script, hosted at
  `https://docs.google.com/spreadsheets/d/11Pst2P18QE3N7lbhnqwdkUN6OONTyTSPNCkSjTz7-RI`
  (rebuild: open sheet → Extensions → Apps Script → paste script → Run → `buildGantt`)

---

## Session Summary — July 6, 2026 (data pipeline session)

- **Decision made:** generator-vs-list question resolved in favor of **Option A**.
  `generate_pairs(number)` now builds and returns a plain list (no `yield`), reasoning:
  9409 pairs (number=97) is small enough that memory efficiency doesn't matter; simplicity wins.
  Implemented as nested loop + `pairs.append(...)` + `return pairs`.
- **`ModularArithmeticDataset` class implemented** in `src/data/modular_arithmetic.py`:
  - Inherits from `torch.utils.data.Dataset` (`from torch.utils.data import Dataset` imported).
  - `__init__(self, number)`: calls `self.pairs = generate_pairs(number)` — stores data once at
    construction so `__len__`/`__getitem__` don't recompute on every call.
  - `__len__`: returns `len(self.pairs)`.
  - `__getitem__(self, idx)`: returns `self.pairs[idx]` (raw tuple `(a, b, target)` for now —
    **not yet split into `(input_tensor, target_tensor)`**; this decision is still open and should
    be revisited once `transformer.py`'s `forward()` signature is designed, since the input format
    depends on whether the model expects two separate numbers or a combined token sequence).
  - `if __name__ == "__main__":` block added for manual testing (`ModularArithmeticDataset(number=10)`,
    prints `len(dataset)`).
- **Debugging exercise (intentional):** Jonathan first wrote the class without `__init__` and without
  inheriting `Dataset`, then deliberately ran it *without* `__init__` to observe the resulting error
  firsthand (learning exercise, not a mistake to fix silently) — confirmed missing `self.pairs`
  causes an `AttributeError` (not an empty list, which is a distinct case he initially guessed).
  This was mentoring-mode teaching, not direct implementation by Claude.

### Still Open / Next Steps

1. **Decide `__getitem__` return shape** — raw tuple (current) vs. `(input_tensor, target_tensor)`
   split. Depends on `transformer.py` `forward()` design (two separate number inputs vs. one
   combined token sequence). Must decide before/while building the model.
2. `get_dataloaders(number)` — split `ModularArithmeticDataset` output + wrap in `DataLoader`
   (not started).
3. `src/models/transformer.py`: token + position embedding → attention → MLP → output head →
   `forward()` (not started).
4. Training loop in `src/train.py` (not started).
5. Run and confirm grokking curve (**M1 gate**) (not started).

---

## Current Project State (Updated — July 6, 2026, later session)

- Scaffold created: `scaffold.py` (project root) generates the full `src/` folder structure —
  `src/__init__.py`, `src/data/__init__.py`, `src/data/modular_arithmetic.py`,
  `src/models/__init__.py`, `src/models/transformer.py`, `src/train.py`. All files existed empty
  before this session's work; `src/data/modular_arithmetic.py` now has real content (below).
- **`generate_pairs(number)` implemented** in `src/data/modular_arithmetic.py`, using nested loops
  and `yield` (generator). Verified correct for `number=97`: 9409 pairs total, each
  `(a, b, (a+b) % number)`.
- Added a `if __name__ == "__main__":` test block that does `pairs = list(generate_pairs(97))` and
  `print(pairs[5])` to manually verify indexing/output — this confirmed the generator produces
  correct values once materialized into a list.
- **Open design question raised but not yet decided:** should `generate_pairs` keep using `yield`
  (generator, not indexable — `ModularArithmeticDataset.__getitem__` would need to convert it via
  `list(...)` internally), or should `generate_pairs` itself build and `return` a list directly
  (Option A) instead of using `yield` (Option B keeps the generator, converts inside the Dataset
  class)? Jonathan understands the generator-vs-list indexing distinction (`pairs[5]` fails on a
  raw generator, works after `list(...)`) but has not yet picked A vs B. **Decide this before
  writing `ModularArithmeticDataset.__getitem__`.**

### What Still Needs to Be Done (Full Rebuild)

1. ~~`generate_pairs(number)` in `src/data/modular_arithmetic.py`~~ — done, Option A (returns list),
   generator-vs-list decision resolved.
2. ~~`ModularArithmeticDataset` class (`__init__`, `__len__`, `__getitem__`)~~ — done, basic version
   working (returns raw tuple from `__getitem__`); return-shape may need revision once
   `transformer.py` input format is decided.
3. `get_dataloaders(number)` — split + wrap in `DataLoader` — next up
4. `src/models/transformer.py`: token + position embedding → attention → MLP → output head → `forward()`
5. Training loop in `src/train.py`
6. Run and confirm grokking curve (**M1 gate**)

---

## Jonathan's Learning Style Notes (carried forward)

- Learning Python while implementing — explain concepts before code; he writes the code himself, Claude only guides when stuck
- Prefers to understand the *purpose* of each function/concept before writing it
- Explains in Hindi/Hinglish when confused — respond in kind
- Very literal thinker — direct logical explanation works better than analogies
- Asks "why does this even exist?" / "agar ye naa karu to kya hoga?" — always answer the negative case directly
- Wants the big picture first when confused ("hamara motive kya hai?") — slow down, one concept at a time
- Does not move on until genuinely understood

---

## Pending (Not Yet Done)

- [ ] Reply to Prof. Rashid's two questions (previous thesis topic + Jammu clarification)
- [x] `generate_pairs` (Option A, list-based)
- [x] `ModularArithmeticDataset` (`__init__`, `__len__`, `__getitem__`) — basic version
- [ ] Decide `__getitem__` return shape (raw tuple vs. tensor split) once model input format is known
- [ ] `get_dataloaders(number)`
- [ ] Rebuild `src/models/transformer.py` (embedding → attention → MLP → output head → `forward()`)
- [ ] Write training loop in `src/train.py`
- [ ] Reproduce canonical Nanda et al. grokking (M1 gate)

---

## Tools & Preferences

| Tool | Preference |
|---|---|
| Spreadsheets | Google Sheets (never Excel) |
| Prompts | Opencode prompt format by default |
| Implementation | Only on explicit request |
