# CLAUDE.md — Grokking Benchmark Project

> **Reserved for Claude only.** Opencode, Antigravity CLI, and any other agents must NOT read, reference, or modify this file.

---

## Project Identity

- **Thesis:** *A Unified Benchmark of Grokking Predictors in Neural Networks*
- **Stack:** Python, PyTorch, MPS (Apple Silicon)
- **Scope:** Head-to-head empirical comparison of 9 published grokking predictors under a unified benchmark protocol

---

## Mandatory Workflow (Every Request)

### Step 1 — Read context.md First

Before anything else, read `context.md` in the project root. It contains session history, confirmed preferences, supervisor instructions, and project decisions that inform all subsequent work.

### Step 2 — Check Graphify

Before reading any source files, check `graphify-out/`:

```
graphify-out/GRAPH_REPORT.md   ← start here
graphify-out/graph.json        ← module relationships
graphify-out/manifest.json     ← file inventory
```

Graphify is the primary source of project understanding. Only read source files when Graphify does not provide enough detail for the specific task. Do **not** scan the entire repo.

### Step 3 — Understand Before Changing

Before proposing or making changes, understand:

- Module structure and dependencies (from Graphify)
- Existing experiment pipeline patterns
- Existing predictor implementations
- Existing data loading and evaluation conventions

Never assume project structure. Never introduce new patterns when existing ones already cover the need.

### Step 4 — Generate an Opencode Prompt (Default)

For all implementation tasks, generate a detailed `.md` prompt for Opencode / Antigravity CLI. Do **not** directly edit files unless the user explicitly says so.

---

## Opencode Prompt Format

Every prompt must include:

```md
IMPORTANT:
Do not read, analyze, reference, or modify CLAUDE.md.
CLAUDE.md is reserved exclusively for Claude.
Use only project source code and existing implementation patterns.

## Objective
## Context
## Relevant Findings
## Files To Inspect
## Requirements
## Constraints
## Implementation Steps
## Validation Steps
## Acceptance Criteria
```

---

## Commit Rule (Mandatory)

Every Opencode / Antigravity CLI prompt must end with an explicit commit instruction:

```md
## Commit
After all changes are validated and passing, run:
  git add .
  git commit -m "<descriptive message>"
  git status

Verify the working tree is clean before finishing.
Do not leave uncommitted files behind.
```

---

## Direct Implementation Override

Claude may directly edit files only when the user explicitly says one of:

- "Implement it yourself"
- "Write the code directly"
- "Edit the files yourself"
- "Claude should make the changes"
- "Do not generate an Opencode prompt"

Without one of these, default to generating an Opencode prompt.

---

## Project-Specific Rules

- **Experiment task:** Reproduce Nanda et al. grokking on `(a+b) mod 97` before any predictor work begins.
- **Predictor order:** L2 Norm → Dropout → Spectral → AGE → HTSR Alpha → Correlation Traps → Weight-PCA → Higher-MI → Commutator Defect
- **Spreadsheets:** Always use Google Sheets. Never Excel/xlsx.
- **Hardware target:** PyTorch MPS (Apple Silicon). Avoid CUDA-only ops.
- **No direct Firebase or external auth connections.**

---

## Non-Negotiable Rules

1. Always read `context.md` before any project analysis.
2. Always check `graphify-out/` before reading any source files.
3. Never scan the full repo — use Graphify to identify relevant files only.
4. Never assume project structure.
5. Generate Opencode prompts by default; direct edits require explicit user instruction.
6. Opencode must never read, reference, or modify `CLAUDE.md`.
7. Every Opencode prompt must include a commit step at the end.
8. Leave the repository in a clean, committed state after every task.
9. If uncertain whether to implement or generate a prompt → generate the prompt.
