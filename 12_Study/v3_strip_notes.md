# Textbook v3 — strip notes (5 Nov 2026)

Source: `Textbook_Common_Concepts_v2.tex` → `Textbook_Common_Concepts_v3.tex`.
Goal (Jonathan, t3u): delete every predictor-specific item; keep only what all predictors share; 5 C's (Clean, Clear, Consistent, Coherent, Chronological).
Same 7 modules / 29 topics and same order as the syllabus. Preamble and macros copied from v2 (one macro dropped, one added; see end).

## Glossary: removed entries
| v2 entry | What it was | v3 |
|---|---|---|
| N21 | L2 norm ‖θ‖₂ and S_t (Nanda's Σw²) | Deleted. Weight-decay confound now says "any quantity computed from the size of the weights" (N20). |
| N32 | Log-epoch grid (L2 smoothing grid, 43.5% before epoch 100, interpolated events like 113.4) | Deleted. N29 just says an event epoch "need not be a whole number". |
| N33 | L2 rules: crossover (50/200), MA-of-MA zero crossing, peak of S_t, halfway reading, y_target | Deleted. Replaced by generic N30 "Rule settings and skip_epochs" and N32 "Illustrative event list". |
| N34 | Dropout 10%-of-rise rule | Deleted (no replacement needed; look-ahead is shown with a generic rule). |
| N35 | skip_epochs "for L2 it's 100" | Folded into generic N30 (no value, no predictor). |
| N47 (part) | "L2 was closed-negative on 26 Sep 2026" | Sentence removed; N45 keeps only "first seven of eight are closed-negative". |
| N54 | "Summary table" = the L2 results table of 26 Sep | Replaced by N35, the generic `seeds_before_grok` count (audit finding 8: counts G−P>0, ignores the 100 margin). |
| N31 (part) | List of the 8 predictor names + "this book uses L2 rules and one Dropout rule" | Names removed; "Your benchmark has eight predictors. This book names none of them." |
| N13 (part) | "doesn't clash with … the event epoch P" (forward reference) | Removed forward reference. |
| N28 (part) | Example signal "S_t or ‖θ_t‖₂", plus y_peak | Example is now the training loss L(θ_t); y_peak dropped (only the L2 halfway rule used it). |
| A10 / reading [3] | Liu et al., Omnigrok (weight norm source) | Removed; citations renumbered [1]–[5] (AdamW is now [3]). |

New glossary: A1–A11 + N1–N51 = **62 entries**, in dependency order (0 forward references; definition-before-use check 0 violations with `tools/def_before_use_v3.py`).

## Worked examples: what replaced each predictor-specific one
| Section | v2 example (removed) | v3 replacement |
|---|---|---|
| 1.2 Memorisation vs generalisation | κ₀ = G₀/P₀ = 12987/113.4 with the L2 crossover; "L2 crossover is an attack point"; self-check citing S_t peak at 1038–1061 | Derivation using only shared G_i: G_min = 6921 > 1000, so acc_test(1000) ≤ 0.9 in all runs → every run is memorising. Self-check: at epoch 8,000 only seed 1 has generalised. |
| 1.3 Circuit formation | Nanda's Σw² falls; your S_t peaks at 1038–1061 | Nanda: cleanup coincides with the acc_test jump, so the circuit exists before it; in your runs the jump is at G_i (6921–23747). |
| 2.4 AdamW | "makes ‖θ_t‖₂ fall"; conclusion "pulls ‖θ_t‖₂ down" | "any signal computed from the size of the weights"; "pulls every weight towards 0". Derivation itself unchanged. |
| 2.5 Epochs | "a predictor … event within 100 epochs" | Same idea, generic wording. Unchanged derivation. |
| 4.1 Predictor and P_i | L2 crossover P = (113.4, 113.2, 120.1, 130.6, 136.8), log-grid, rounding | **Illustrative List A** P = (3500, 6850, 2900, 4100, 7800), "suppose some rule fires at…". Viva answer made generic. |
| 4.2 Lead | Seed 0 of L2 crossover (113.4 < 12887); self-check about halfway reading, seed 2 by 13 epochs | Seeds 0 and 1 of List A: seed 0 leads; seed 1 (6850) is before grok but inside the margin (6850 > 6821). Self-check uses seeds_before_grok on List A seed 1. |
| 4.3 Lead fraction | L2 crossover 5/5 = 1.0 | List A: 4/5 = 0.8 (exactly on the threshold). |
| 4.4 Relative lead | L2 crossover r₀ = 0.9913, r̄ = 0.989 | List A: r₀ = 9487/12987 ≈ 0.7305; r̄ = 0.547 (S = 2.26472). Self-check "halfway to grok" reworded "at half the grok epoch". |
| 4.5 Misses | generic (kept) | unchanged; "possible attack point for every predictor". |
| 5.1 Spearman ρ | L2 crossover ρ = 0.70 | List A: R(P) = (2,4,1,3,5), d = (2,−3,1,0,0), Σd² = 14, ρ = 0.3. |
| 5.2 Pearson vs Spearman | L2 crossover rounded to whole epochs → tie → ρ = 0.564; viva answer with L2 final-norm Pearson −0.55 / +0.96 | **Illustrative List B** (4000, 2000, 3000, 4000, 9000) with a real tie: ρ = 9.5/√95 ≈ 0.975 (between the untied 1 and 0.9). Viva answer generic (one extreme seed can decide Pearson, even its sign). |
| 5.3 21 values | generic (kept) | unchanged; cross-check line with List A (D = 14 → 0.3). |
| 5.4 Significance | "ρ = 0.70 isn't significant" (L2) | "List A's ρ = 0.3 is easy to get by chance"; 14/120 orderings give ρ ≥ 0.7 kept (verified). Rounding of 1/24 now says "rounded" (0.0417 → 0.042), not "dropped". |
| 6.1 Frozen protocol | "written after L2 closed-negative on 26 Sep"; self-check about L2 left out of the scored list on 2 Oct | "Any result already known by 29 Sep weakens the blind-threshold claim; check dates in context.md." Self-check: does every predictor have a saved verdict? |
| 6.2 PASS rule | L2 crossover verdict + L2 MA-of-MA derivation (ρ = 0, r̄ = 0.835) | Summary of List A (0.8 ✓, 0.3 ✗, 0.547 ✓ → FAIL) + derivation on **illustrative List C** (2020, 1990, 2000, 2030, 2010), a near-fixed clock: lead 1.0, ρ = 0.6, r̄ = 0.816 → FAIL. |
| 6.3 Pre-registration | "after L2's results were known"; fixed control at P = 1800 | "after some results were already known"; fixed control at P = 2000 (1800 echoed the MA-of-MA epochs). |
| 7.1 Hindsight | L2 halfway-down-from-peak (y_target = 0.5 y_peak + 0.5 y_final), Dropout 10% rule, halfway PASS numbers | Generic illustrative rule "fires when y_t falls below its whole-run average"; derivation splits ȳ_run into past + future sums (40,000 − τ future terms). |
| 7.2 Causal vs centred | "both L2 rules use uniform_filter1d"; "viva notes call the MA crossover ex-ante" | uniform_filter1d stated as a centred function; "a rule that smooths with it can't be called ex-ante". Derivation unchanged. |
| 7.3 Researcher degrees of freedom | L2 skip_epochs 200 → 100; crossings at epochs 2, 3, 42 | Illustrative 3 windows × 2 thresholds × 2 skip values = 12 versions; with 1/24 per version (Section 5.4) and an explicit independence caveat, P(at least one ρ ≥ 0.9) = 1 − (23/24)¹² ≈ 0.400. |
| 7.4 Sensitivity | L2 peak event r̄ = 0.904, misses by 0.004 | Illustrative P_i = 0.09 G_i: lead 1.0, ρ = 1, r̄ = 0.91 → misses by 0.01; a 0.92 limit would pass it. |
| 7.5 Baselines | L2 noise floor saved but unused, measured before epoch 90 on the log grid; λ = 0 shows S_t | Generic: a noise floor only helps if a rule uses it; λ = 0 run shows what any y_t does without grokking. |
| 7.6 Negative results | "L2 events recomputed match summary.json" | Sentence removed; G_i and split reproducibility kept. |

## Kept (shared) facts — unchanged
p = 113; modular addition; 30% split hard-coded (3830/8939, audit finding); 5 seeds; 40,000 epochs; checkpoints every 50, evaluation every 100; AdamW β = (0.9, 0.98), λ = 1.0; W_O gap; grok threshold 0.9; G = (12987, 6921, 9510, 11019, 23747); lead margin 100; lead_frac with misses dropped (4/4); n < 4 auto-fail; PASS rule incl. 0.8999999999999998; Spearman/Pearson/ties; 21 values; 1/120 and 1/24; context.md "11 values" error; frozen protocol date 29 Sep 2026; seven of eight closed-negative; criterion validation with P = 0.5 G.

## Preamble / macro changes
- Dropped `\ypeak`, `\ytarget` (only the L2 halfway rule used them).
- Added `\yrun` (whole-run average, Section 7.1 only).
- Everything else (XeLaTeX, A4, 2.2 cm, 11pt, DejaVu Serif / Sans Mono 0.85, parindent 0, parskip 6pt, Module headings, \gl \fdef \fwhy \fex \fself \fviva \dass \dgiven \dlet \dfind \dsol \why, dlist/glist) identical to v2.

## Checks run
- `rg 'L2|Dropout|Spectral|AGE|HTSR|Weight-PCA|Higher-MI|Commutator|crossover|MA-of-MA|halfway'` → 0 hits. Case-insensitive search for l2|dropout|spectral|htsr|pca|commutator|crossover|halfway|omnigrok|norm|S_t|log-epoch → 0 hits.
- Glossary: 62 entries, 0 forward references, no `\gl{}` pointing to a missing entry.
- `tools/def_before_use_v3.py` (v3 numbering): 0 violations.
- Syntax-only check `xelatex -no-pdf` (no PDF made): no errors, no undefined control sequences, no overfull boxes; 43 pages in the .xdv. PDF still to be built by a Cursor cloud agent.
- All worked-example numbers recomputed in Python (ranks, ρ, r̄ sums at 5 d.p., (23/24)^12, 14/120, 21 D-values).
