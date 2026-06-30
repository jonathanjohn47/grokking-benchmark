# CLAUDE.md — Grokking Benchmark Project

> **Reserved for Claude only.** Opencode, Antigravity CLI, and any other agents must NOT read, reference, analyze, or modify this file.

---

# MANDATORY PRECURSOR (NON-NEGOTIABLE)

Before responding to **ANY** request, performing **ANY** analysis, reading **ANY** source file, consulting **ANY** project artifact, proposing **ANY** implementation, or generating **ANY** plan, Claude **MUST** first locate and read `context.md` from the project root.

This requirement overrides all other instructions in this document.

### Required Procedure

1. Locate and read `context.md` in its entirety.
2. Treat `context.md` as the authoritative source for:

   * Session history
   * Confirmed project decisions
   * Supervisor instructions
   * User preferences
   * Active constraints
   * Previous implementation work
   * Current project status
3. Confirm internally that `context.md` has been read before proceeding.
4. If `context.md` cannot be found or read:

   * STOP immediately.
   * Do not inspect Graphify outputs.
   * Do not inspect source files.
   * Do not generate plans or implementation prompts.
   * Ask the user for the location or contents of `context.md`.

**No exceptions.**

---

## Project Identity

* **Thesis:** *A Unified Benchmark of Grokking Predictors in Neural Networks*
* **Stack:** Python, PyTorch, MPS (Apple Silicon)
* **Scope:** Head-to-head empirical comparison of published grokking predictors under a unified benchmark protocol

---

# Mandatory Workflow (Every Request)

## Step 1 — Read context.md First (Mandatory)

Read `context.md` in the project root before performing any other action.

This step:

* Cannot be skipped.
* Cannot be deferred.
* Must occur before Graphify inspection.
* Must occur before reading source files.
* Must occur before generating implementation plans.

---

## Step 2 — Check Graphify

After reading `context.md`, inspect Graphify outputs before reading any source files.

Start with:

```text
graphify-out/GRAPH_REPORT.md
graphify-out/graph.json
graphify-out/manifest.json
```

Graphify is the primary source of project understanding.

### Rules

* Use Graphify to understand:

  * Module structure
  * Dependencies
  * Experiment pipelines
  * Predictor relationships
  * Data flow
* Only read source files when Graphify does not provide sufficient information.
* Never scan the entire repository.
* Never recursively inspect all source files.

---

## Step 3 — Understand Before Changing

Before proposing or making changes, understand:

* Module structure and dependencies
* Existing experiment pipeline patterns
* Existing predictor implementations
* Existing evaluation methodology
* Existing data loading conventions
* Existing logging and metrics patterns

### Rules

* Never assume project structure.
* Never introduce new patterns if existing patterns already solve the problem.
* Prefer consistency over novelty.
* Reuse existing abstractions whenever possible.

---

## Step 4 — Generate an Opencode Prompt (Default)

For all implementation tasks:

* Generate a detailed `.md` implementation prompt for Opencode or Antigravity CLI.
* Do not directly edit project files unless the user explicitly instructs Claude to do so.

---

# Opencode Prompt Template

Every generated prompt must begin with:

```md
IMPORTANT:
Do not read, analyze, reference, or modify CLAUDE.md.
CLAUDE.md is reserved exclusively for Claude.
Use only project source code and existing implementation patterns.
```

Every prompt must include the following sections:

```md
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

# Commit Rule (Mandatory)

Every Opencode / Antigravity CLI prompt must end with:

```md
## Commit

After all changes are validated and passing, run:

git add .
git commit -m "<descriptive message>"
git status

Verify the working tree is clean before finishing.
Do not leave uncommitted files behind.
```

This requirement is mandatory.

---

# Direct Implementation Override

Claude may directly modify files only if the user explicitly says one of the following:

* "Implement it yourself"
* "Write the code directly"
* "Edit the files yourself"
* "Claude should make the changes"
* "Do not generate an Opencode prompt"

Without one of these explicit instructions, Claude must generate an Opencode prompt.

---

# Project-Specific Rules

## Experiment Prerequisite

Before implementing predictor benchmarks:

* Reproduce Nanda et al. grokking on:

```text
(a+b) mod 97
```

No predictor work begins until reproduction succeeds.

---

## Predictor Evaluation Order

Evaluate predictors in the following order:

1. L2 Norm
2. Dropout
3. Spectral
4. AGE
5. HTSR Alpha
6. Correlation Traps
7. Weight-PCA
8. Higher-MI
9. Commutator Defect

---

## Spreadsheet Policy

* Always use Google Sheets.
* Never use Excel or `.xlsx` files.

---

## Hardware Target

Target hardware:

```text
PyTorch + Apple Silicon MPS
```

Rules:

* Avoid CUDA-only operations.
* Ensure MPS compatibility.
* Prefer portable PyTorch APIs.

---

## External Services

Prohibited:

* Direct Firebase connections
* External authentication providers
* Production credentials
* Secret management systems

---

# Non-Negotiable Rules

1. Always read `context.md` before any project analysis.
2. If `context.md` cannot be read, stop immediately.
3. Always inspect Graphify before reading source files.
4. Never scan the entire repository.
5. Never assume project structure.
6. Generate Opencode prompts by default.
7. Direct implementation requires explicit user authorization.
8. Opencode must never read or modify `CLAUDE.md`.
9. Every Opencode prompt must include a commit step.
10. Leave the repository in a clean, committed state.
11. Reuse existing project patterns whenever possible.
12. If uncertain whether to implement or generate a prompt, generate a prompt.
13. `context.md` is the authoritative source of project state.
14. The `context.md` requirement overrides all other instructions.
