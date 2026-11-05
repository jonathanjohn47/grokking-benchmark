# Textbook v4: expansion notes (5 Nov 2026)

Source: `Textbook_Common_Concepts_v3.tex` → `Textbook_Common_Concepts_v4.tex` (built by `tools/build_v4.py` from the prose in `tools/v4_content.py`).
Goal (Jonathan, t5u): v3 sentences were too thin. Keep the H.K. Dass layout, the 5 C's, and no predictor-specific content, but explain and elaborate much more.

## What stayed exactly as in v3
- Preamble, fonts, margins (XeLaTeX, A4, 2.2 cm, 11pt, DejaVu Serif / Sans Mono 0.85, parindent 0, parskip 6pt).
- Same 7 modules / 29 sections, same order.
- Glossary: same 62 entries (A1–A11, N1–N51), same wording, same numbering, dependency order. Only the glossary intro got two more sentences (read it once, then use it as a reference).
- Every Dass derivation (Given / Let / To find / Solution / ∴ / boxed) unchanged: same numbers, same steps.
- Illustrative Lists A, B, C and the one-line lists (0.5 G, 0.09 G, whole-run-average rule, 12-version count) unchanged.
- All v3 Definition and Why-it-matters sentences kept word for word; all viva questions kept.

## What was added
New labels (macros): `\fplain` = **In plain words:**, `\fread` = **Why it matters and how to read it:** (replaces `\fwhy` in the body; `\fwhy` still defined), `\fans` = **Reading the answer:**.

Every section now reads: In plain words → Definition → Why it matters and how to read it (v3 text + a new intuition paragraph) → Worked example (Dass) → Reading the answer (where there's a result to interpret) → Self-check → Viva.

- **Title page:** one paragraph explaining the topic layout.
- **Module intros (7):** a paragraph after each module heading saying what the module does and how its sections connect (chronology).
- **In plain words (29/29 sections):** what the idea is, why it's in the thesis, what mistake it prevents.
- **How to read it (29/29):** a second paragraph of intuition built only from setup objects (pairs, logits, loss, gradient, weights, accuracy, epochs, seeds, ranks).
- **Reading the answer (14 sections):** 1.2, 2.3, 2.4, 3.2, 4.2, 4.3, 4.4, 4.5, 5.1, 5.4, 6.2, 7.1, 7.3, 7.4; says what the boxed number means.
- **Self-check answers expanded (13):** 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.5, 2.6, 3.1, 3.2, 4.5, 5.3, 7.5 (7.5 had no answer in v3; now has a short one).
- **4.1:** two sentences pointing out List A's features (seed 1 near grok; order mismatch) that 4.2 and 5.1 use.

Size: body prose (non-derivation lines) ~5,000 words in v3 → ~12,100 in v4 (~2.4×). `xelatex -no-pdf`: 43 → 55 pages.

## New numbers in prose (all derived from existing facts, checked)
- 1/8939 ≈ 0.0001 (one test pair's weight in acc_test), Section 3.1.
- 23747/6921 ≈ 3.4 (latest vs earliest grok), Section 3.2.
- 12836.8/40000 ≈ a third (mean grok vs run length), Section 2.6.
- 23747 is "nearly twice" 12987 (1.83), Section 5.2.
- Seed 1 of List A misses the cutoff by 6850 − 6821 = 29, Section 4.2.
- List A per-seed r between ≈0.63 and 0.73 (seed 1 ≈ 0.01), Section 4.4.
- List A's Σd² = 14: 9 from seed 1, 4 from seed 0, Section 5.1.
- List C spans 1,990–2,030 (40 epochs) vs G span 16,826, Section 6.2.
- 1/24 ≈ 4%; 12 versions → ≈40% (≈10×), Sections 5.4, 7.3.
- With 5 firing seeds lead_frac ∈ {0, 0.2, …, 1}, Section 4.3. With n = 5, ρ ≥ 0.9 ⇔ D ≤ 2, Section 5.3.

No new scientific claims: the intuition paragraphs restate v3 facts (Nanda's three phases, weight-decay confound, centred vs causal, look-ahead) in more words. No analogies invoking new mechanisms.

## Checks run
- `rg 'L2|Dropout|Spectral|AGE|HTSR|Weight-PCA|Higher-MI|Commutator|crossover|MA-of-MA|halfway|Omnigrok'` → 0 hits.
- Case-insensitive `l2|dropout|spectral|htsr|pca|commutator|crossover|halfway|omnigrok|norm|log-epoch|kappa` → 0 hits.
- `tools/def_before_use_v3.py` on v4 → 0 violations (one hit on the word "memorise" in the glossary intro was reworded).
- `xelatex -no-pdf` ×2 (in /tmp, no PDF made): no errors, no undefined control sequences, no overfull boxes, no missing characters; 55 pages in the .xdv. Only the italic-shape font warning that v3 also has.
