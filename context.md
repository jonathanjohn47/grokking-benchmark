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

## Session Summary — July 6, 2026 (get_dataloaders + code review session)

- **`get_dataloaders(number, batch_size=32)` implemented** in `src/data/modular_arithmetic.py`
  (same file as `generate_pairs`/`ModularArithmeticDataset` — kept together since it's all part of
  the same data pipeline; `train.py` will import it from there).
  - Uses `torch.utils.data.random_split` to split the full `ModularArithmeticDataset` into
    train/test, then wraps each half in a `DataLoader`.
  - **Bug found and fixed:** initial version called `torch.utils.data.random_split(...)` /
    `torch.utils.data.DataLoader(...)` without importing `torch` itself (only
    `from torch.utils.data import Dataset` was imported) — would have raised `NameError`.
    Fixed by importing `Dataset, DataLoader, random_split` directly from `torch.utils.data`.
  - **Design choice flagged (not yet confirmed as intentional):** `train_size = int(0.3 * len(pairs))`
    — only 30% train / 70% test split. This may be intentional for grokking (small train fraction is
    part of what causes the grokking phenomenon — model memorizes first, generalizes later), but
    Jonathan has not yet explicitly confirmed the reasoning. **Revisit before M1 gate run.**
  - **Still open:** `train_dataloader` does not set `shuffle=True`. Needs decision — test loader
    correctly has no shuffle, but train loader should very likely shuffle each epoch.
  - **Still open (minor):** variable name `pairs = ModularArithmeticDataset(number)` inside
    `get_dataloaders` is misleading — it holds a `Dataset` object, not a list of pairs. Rename
    suggested but not yet applied.
- **`ModularArithmeticDataset` no longer inherits `Dataset`** (Jonathan removed
  `class ModularArithmeticDataset(Dataset):` → `class ModularArithmeticDataset:` as a deliberate
  experiment to see what breaks — same debugging-by-experiment learning style as before).
  - Confirmed via mentoring (not yet re-tested by running code) that this **will NOT break**
    `DataLoader`/`random_split` at runtime, because both rely on duck typing (`len()` +
    `__getitem__`), not `isinstance(dataset, Dataset)` checks. The one `isinstance` check
    (`IterableDataset`) is a negative check that still passes correctly either way.
  - What IS lost by not inheriting: `Dataset.__add__` (no `dataset1 + dataset2` via `ConcatDataset`),
    type-checker/IDE clarity, and the explicit "this is a PyTorch Dataset" contract.
  - **Recommendation given:** re-inherit `Dataset` as best practice even though nothing currently
    breaks. Jonathan has not yet made this change back.
- **Concepts taught this session (mentor mode):** what `DataLoader` is and why it's needed (batching,
  shuffling, iteration boilerplate removal); why inheriting from `Dataset` matters for PyTorch's
  data-loading ecosystem; how to print all list items without an explicit `for` loop
  (`print(*pairs, sep="\n")` via unpacking).

### Still Open / Next Steps (updated)

1. Confirm/fix `train_size` ratio in `get_dataloaders` (0.3 — intentional for grokking or mistake?).
2. Add `shuffle=True` to `train_dataloader` in `get_dataloaders`.
3. Rename misleading `pairs` variable inside `get_dataloaders` (holds a Dataset, not a list).
4. Decide whether to re-inherit `Dataset` in `ModularArithmeticDataset` (recommended, not yet done).
5. Decide `__getitem__` return shape (raw tuple vs. tensor split) — still open, depends on
   `transformer.py` input format.
6. Build `src/models/transformer.py` (not started).
7. Write training loop in `src/train.py` (not started).
8. Run and confirm grokking curve (**M1 gate**) (not started).

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

## Session Summary — July 6, 2026 (data pipeline code review, close-out)

- **Reviewed final state of `src/data/modular_arithmetic.py`** — all previously open issues from
  the last session resolved this session:
  1. **Train/test split ratio (0.3) confirmed correct** — checked against the attached Obsidian
     vault (`Grokking Master Thesis`). Both `03 - Literature/Grokking: Generalisation Beyond
     Overfitting on Small Algorithmic Datasets.md` ("Training split | ~30% train, ~70% test") and
     the thesis's own `05 - Thesis/Experimental Designs Used in Literature.md` factorial design
     table document 30%/70% as the standard split for grokking experiments (small train fraction
     is intentional — it's what causes the memorize-first-generalize-later dynamic). **Not a bug,
     no change needed.**
  2. **`test_dataloader` shuffle fixed** — Jonathan removed `shuffle=True` from the test
     `DataLoader` (now defaults to `False`), correct since shuffling eval data has no benefit.
  3. **Misleading variable renamed** — `pairs = ModularArithmeticDataset(number)` inside
     `get_dataloaders` renamed to `modular_arithmetic_dataset` for clarity (first to
     `modularArithmeticDataset`, then corrected to Python's `snake_case` convention on Claude's
     suggestion).
  4. **`ModularArithmeticDataset(Dataset)` inheritance restored** (was removed in the prior
     session as a deliberate experiment; confirmed back in place this session).
- **Concept taught (mentor mode):** clarified why `ModularArithmeticDataset(number)` returns an
  **instance of the class**, not a list — `ClassName(...)` always invokes `__init__` and returns
  a new object, even though `__init__` internally builds `self.pairs` from a list. `__init__`
  itself never returns anything; the object holds the list as an attribute. This is why the old
  variable name `pairs` was misleading (it held a `Dataset` object, not a list).
- **Current state of `src/data/modular_arithmetic.py` (verified correct, no open issues):**
  ```python
  from torch.utils.data import Dataset, DataLoader, random_split

  def generate_pairs(number):
      pairs = []
      for i in range(number):
          for j in range(number):
              pairs.append((i, j, (i + j) % number))
      return pairs


  def get_dataloaders(number, batch_size=32):
      modular_arithmetic_dataset = ModularArithmeticDataset(number)
      train_size = int(0.3 * len(modular_arithmetic_dataset))
      test_size = len(modular_arithmetic_dataset) - train_size

      train_dataset, test_dataset = random_split(modular_arithmetic_dataset, [train_size, test_size])
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

      return train_dataloader, test_dataloader


  class ModularArithmeticDataset(Dataset):
      def __init__(self, number):
          self.pairs = generate_pairs(number)

      def __len__(self):
          return len(self.pairs)

      def __getitem__(self, idx):
          return self.pairs[idx]
  ```
- **Data pipeline is now considered fully done and closed out.** Next up per the project plan is
  `src/models/transformer.py`.

### Still Open / Next Steps (updated)

1. Decide `__getitem__` return shape (raw tuple vs. `(input_tensor, target_tensor)` tensor split)
   — still open, depends on `transformer.py`'s `forward()` input format (two separate number
   inputs vs. one combined token sequence). **Decide this while designing the model.**
2. Build `src/models/transformer.py`: token + position embedding → attention → MLP → output head
   → `forward()` (not started — this is the current next action).
3. Write training loop in `src/train.py` (not started).
4. Run and confirm grokking curve (**M1 gate**) (not started).
5. Reply to Prof. Rashid's two open questions (previous thesis topic + Jammu clarification) —
   still pending, unrelated to the code track.

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
- [x] `ModularArithmeticDataset` (`__init__`, `__len__`, `__getitem__`, inherits `Dataset`) — done
- [ ] Decide `__getitem__` return shape (raw tuple vs. tensor split) once model input format is known
- [x] `get_dataloaders(number)` — fully done: 0.3 train ratio confirmed correct against literature/
      vault, `shuffle=True` on train loader only, variable renamed (`modular_arithmetic_dataset`,
      snake_case), `Dataset` inheritance restored. **Data pipeline closed out, no open issues.**
- [ ] Rebuild `src/models/transformer.py` (embedding → attention → MLP → output head → `forward()`)
      — **current next action**
- [ ] Write training loop in `src/train.py`
- [ ] Reproduce canonical Nanda et al. grokking (M1 gate)

---

## Tools & Preferences

| Tool | Preference |
|---|---|
| Spreadsheets | Google Sheets (never Excel) |
| Prompts | Opencode prompt format by default |
| Implementation | Only on explicit request |
