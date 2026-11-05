# Audit of the original Common Concepts textbook (25 pages, pdfTeX)

Checked against Jonathan's standards, the syllabus material, CHAT_SUMMARY.md and STYLE_GUIDE.md. Page numbers are physical PDF pages of `original.pdf`. Quotes are from the PDF text (subscripts appear flattened, e.g. `θk` for θ_k).

Fixes refer to `Textbook_Common_Concepts_v2.tex`: glossary entries A1–A11 and N1–N54, and sections 1.1–7.6.

## Count by type

| Type | Count |
|---|---|
| Used before defined | 14 |
| New term out of nowhere | 30 |
| Inconsistent symbol / type | 17 |
| Collapsed or missing step | 10 |
| Wrong or unverified fact | 15 |
| Layout | 10 |
| Dass-layout or equation-rule breach | 9 |
| Style / filler | 4 |
| **Total** | **109** |

Two types were added to the six requested: *Dass-layout or equation-rule breach* (Given without source, Let without type/evaluation/sign/role, = for rounded values, arrows instead of equations, missing ∴) and *Style / filler*. Problems that repeat in every Dass block are listed once, at their first page, and say so.

## Worst five

1. **p.23–24, Appendix A, `ηO(η²) = O(η²)` and the `≈` line** (wrong fact): the remainder should be O(η³); as written it swallows the η² term the boxed result claims to keep, and the ≈ line drops that term while the box keeps it.
2. **p.15–16, `lead frac = #{P < G − 100}/5` vs `lead frac = 4/4`** (inconsistent definition): the denominator switches from 5 to the number of firing seeds between 4.3 and 4.5.
3. **p.7 / p.19, L2 crossover and MA-of-MA** (new terms out of nowhere): the two L2 rules every example is built on are never defined anywhere in the book.
4. **p.19, `fixed P = 1800 ρ = 0`** (wrong fact): a constant event list has all ranks tied, so ρ is undefined, not 0.
5. **p.5, `(for example at θk)`, `g_A points uphill`** (used before defined): the reading guide uses symbols first defined 18 pages later in the Appendix, and there's no glossary at all.

## All problems

| # | Page | Quote | Type | Problem | Fix in v2 |
|---|---|---|---|---|---|
| 1 | 1 | `p = 113, 40,000 epochs, 5 seeds: 12987, 6921, 9510, 11019, 23747` | Wrong or unverified fact | The five numbers are grok epochs, not seeds. Seeds are 0 to 4. | Removed from the title page. Seeds are N8 (i = 0..4); grok epochs are N27. |
| 2 | 1 | `(whole book) pdfTeX, URW Palladio fonts` | Layout | STYLE_GUIDE asks for XeLaTeX with DejaVu Serif / DejaVu Sans Mono. | v2 uses XeLaTeX, fontspec, DejaVu Serif, DejaVu Sans Mono (Scale 0.85), A4, 2.2 cm margins, 0 pt indent, 6 pt paragraph skip. |
| 3 | 2 | `(blank page)` | Layout | Blank pages 2, 6, 8, 12, 14 and 20 (two-sided openright layout). | oneside, openany: no blank pages. |
| 4 | 3 | `1.1 1.1 Grokking ... 7.6 7.6 Negative Results` | Layout | Every one of the 29 section numbers is doubled in the contents and in the headings, because the number was typed into the title and LaTeX numbered it again. | Titles carry no typed numbers; LaTeX numbers them once (1.1 Grokking). |
| 5 | 3 | `Contents (no Appendix A entry)` | Layout | Appendix A isn't in the contents. | Appendix removed from this book (see row on Appendix scope). |
| 6 | 5 | `(whole book) no notation list` | New term out of nowhere | There is no glossary or notation list, so terms appear wherever they happen to be used first. | New "Notation and glossary" chapter first: 65 entries (A1-A11, N1-N54) in dependency order, each giving what, type, where evaluated, sign/direction, role. |
| 7 | 5 | `(for example at θk)` | Used before defined | θ_k used on page 5; first defined on page 23 (Appendix). | θ_t defined in N10 before any use; one time index t throughout. |
| 8 | 5 | `g A points uphill, so −ηg A is one update downhill` | Used before defined | g_A and η used on page 5; defined only in the Appendix, page 23. | Gradient g(θ_t) (N16) and η (N17) defined in the glossary; no batch-A gradient in this book. |
| 9 | 5 | `+O(η 2 )), and O(η n ) defined in Let block first time` | New term out of nowhere | Big-O remainder notation introduced in the reading guide although no main-text derivation uses it. | Dropped; the reading guide (A5) now says what = and ≈ mean. |
| 10 | 5 | `Fixed vocabulary: loss, batch, examples, weights θ, `i , gX , HX , η.` | Inconsistent symbol / type | The "fixed vocabulary" is listed but not defined; ℓ_i never appears again (ℓ, ℓ_X and "loss = −ln softmax(z)_correct" are used instead); "examples" is never used; it omits nearly every symbol the book does use (G, P, ρ, r̄, lead frac, d_i, n...). | Replaced by the full glossary; the body uses only glossary names. |
| 11 | 7 | `Nanda-Unified: training hits 100% within few hundred epochs` | New term out of nowhere | "Nanda-Unified" is an undefined name. | Says "your runs"; the setup file is A9 (nanda_unified.yaml). |
| 12 | 7 | `Mean ≈ 12836.` | Wrong or unverified fact | The mean is 64184/5 = 12836.8 exactly; written as ≈ 12836 it is both truncated and marked approximate. | Section 2.6 derives Ḡ = 12836.8 in Dass layout. |
| 13 | 7 | `G = min{t : acctest (t) > 0.9}` | Used before defined | G and acc_test used in 1.1; defined in 3.1/3.2. | acc_test is N23, G_i is N27, both before Module 1. |
| 14 | 7 | `occurring long after acctrain ≈ 1` | Used before defined | acc_train used in 1.1 and never defined anywhere. | acc_train defined in N23. |
| 15 | 7 | `Slow learning rises together; grokking has long flat gap.` | Style / filler | Fragment; "slow learning" is undefined. | Full sentences: "In slow learning acc_train and acc_test rise together..." |
| 16 | 7 | `L2 crossover fires at 113–137, memorisation phase` | New term out of nowhere | "L2" and "crossover" are never defined anywhere in the book. | L2 Norm predictor and its crossover rule (windows 50/200, centred, log-epoch grid, first 100 epochs skipped) defined in N31-N33. |
| 17 | 7 | `about 100 times earlier than grok — attack point.` | New term out of nowhere | "attack point" never defined. | Defined in A11 (with viva and examiner). |
| 18 | 7 | `about 100 times earlier than grok` | Collapsed or missing step | Ratio stated with no calculation; per seed it is 61 to 174 times. | Section 1.2 Dass derivation κ_0 = 12987/113.4 = 114.5, plus the other four seeds. |
| 19 | 7 | `∑ w2 falls smoothly during circuit formation` | Inconsistent symbol / type | w is never defined; the book calls the weights θ elsewhere. | S_t = \|\|θ_t\|\|_2^2 (N21); N21 notes that Nanda writes it as Σw². |
| 20 | 7 | `circuit formation (general algorithm built quietly)` | New term out of nowhere | "circuit" undefined. | N26 defines circuit and the three phases. |
| 21 | 7 | `(all 29 topics)` | Layout | No topic has a Self-check, and only 1.1 has a Likely viva question; several topics lack "Why it matters" or a worked example. | Every topic has Definition, Why it matters, Worked example, Self-check (with short answers where the sources give them) and Likely viva question. |
| 22 | 9 | `For a = 100, b = 50: (150) mod 113 = 37.` | Collapsed or missing step | Skips 100 + 50 = 150 and the remainder step 150 − 113. | Section 2.1 Dass block: addition, then 150 − 1×113, then 37, one line each. |
| 23 | 9 | `Answers: p = 113 (0 to 112).` | Inconsistent symbol / type | p is the modulus; here it is also used as "number of answers". | Answers are k in {0,...,112} (N11); p stays the modulus. |
| 24 | 9 | `With “=” token, vocab = 114.` | Wrong or unverified fact | "token" and "vocab" undefined, and vocab = 114 is not in your source files. | Dropped. |
| 25 | 9 | `config train fraction does not change split` | New term out of nowhere | "config" undefined at first use. | A9 names nanda_unified.yaml as the configuration (config) file; N6 holds the train_fraction finding. |
| 26 | 9 | `H.K. Dass Derivation: Train/Test Split      Given:` | Layout | "Given:" runs into the heading line (same in all 7 Dass blocks). | Heading, Given, Let, To find and Solution each start on their own line. |
| 27 | 9 | `2. Total pairs N = p2 .` | Dass-layout or equation-rule breach | Given lines don't say where they come from (all Dass blocks). | Every Given line cites a glossary entry, a section, or your file. |
| 28 | 9 | `1. p: scalar, prime modulus ... 4. Ntrain : scalar, b f N c.` | Dass-layout or equation-rule breach | Integers labelled "scalar" (p, N, N_train, N_test; later c, n, ranks), and Let lines lack where evaluated, sign/direction and role (all Dass blocks). | Let lines give type (integer / scalar / vector with M or 113 entries / set), where evaluated, sign, role. |
| 29 | 10 | `One-layer, four-head transformer outputs 113 scores (logits)` | New term out of nowhere | "transformer", "head" and "logits" appear without definition. | N5 (model, weights, training) and N11 (logits). |
| 30 | 10 | `loss = − ln softmax(z)correct` | Inconsistent symbol / type | Same loss written two ways: −ln softmax(z)_correct here and ℓ = −ln p_c in the derivation. | One form: ℓ = −ln q_c (N14). |
| 31 | 10 | `softmax(z)k = ezk / ∑ j ez j` | Used before defined | z, k and j used without definition. | N11 (z, k), N13 (q_k, j). |
| 32 | 10 | `If correct prob 0.25, loss = 1.386; at 0.99 loss ≈ 0.01.` | Dass-layout or equation-rule breach | = used for a rounded value (−ln 0.25 = 1.3863...). | Section 2.3 Dass block with exact lines and the rounded value only in the box. |
| 33 | 10 | `3. pk : scalar, softmax(z)k .` | Inconsistent symbol / type | p_k (probability) clashes with the modulus p and the event P. | Probabilities are q_k (N13). |
| 34 | 10 | `2. Correct index c: scalar.` | Inconsistent symbol / type | c is an integer class index, not a generic scalar. | c is an integer in {0,...,112} (N3). |
| 35 | 11 | `p0c = ... `0 = − ln p0c` | Used before defined | p'_c and ℓ' are used in the Solution but not in the Let list. | Let list defines s, z', q'_c and ℓ' before the Solution. |
| 36 | 11 | `∵ e a scalar pull out ∵ a scalar` | Layout | Garbled reason with a doubled "because". | "e^s is the same scalar in every term, so it comes out of the sum". |
| 37 | 11 | `= e a e zc / ∑ j ea e j  ∵ e x +y = e x ey` | Collapsed or missing step | The exponent rule is applied to the numerator and every denominator term in one line. | Two lines: numerator, then denominator. |
| 38 | 11 | ``0 = ` (boxed), no ∴ line` | Dass-layout or equation-rule breach | No ∴ conclusion in plain words (same in 4.4, 5.1, 7.1, 7.2 blocks). | Every derivation ends with a ∴ sentence and a boxed answer. |
| 39 | 11 | `Gap: no attention output matrix WO vs Nanda.` | New term out of nowhere | W_O and "attention" appear out of nowhere, dropped after an unrelated derivation. | N5 defines W_O as the known gap; used in the 2.3 viva answer. |
| 40 | 11 | `AdamW optimiser, weight decay λ = 1.0, betas (0.9, 0.98).` | New term out of nowhere | "optimiser", λ and "betas" undefined. | N18 (AdamW, β1, β2, u_t), N19 (λ). |
| 41 | 11 | `Decoupled: w ← w − ηλw plus gradient step.` | Inconsistent symbol / type | Update written with an arrow, not an equation; uses w instead of θ; η undefined; "decoupled" undefined. | Equation θ_{t+1} = θ_t − η u_t − ηλ θ_t (N20); Section 2.4 Dass block derives (1 − η)θ_t − η u_t. |
| 42 | 11 | `Confound: falling norm could be optimiser not grokking.` | New term out of nowhere | "confound" and "norm" undefined. | N20 defines confound; N21 defines the norm. |
| 43 | 11 | `Save every 50, eval every 100. Event resolution 100 epochs.` | Used before defined | "Event" used before it is defined (4.1). | Event epoch is N30, before Module 2. |
| 44 | 11 | `Event resolution 100 epochs.` | Wrong or unverified fact | Contradicts your own numbers: G = 12987 and P = 113.4 are not on a 100-epoch grid. | 2.5 explains: checkpoint evaluations are 100 apart, but G comes from a finer accuracy curve and L2 events from the log-epoch grid (N32); asks you to check which curve G is read from. |
| 45 | 11 | `Frozen protocol scores 5 seeds.` | Used before defined | "Frozen protocol" used in 2.6; defined in 6.1. | N46, before the modules. |
| 46 | 11 | `per predictor min runs:3 but scored are five` | Wrong or unverified fact | "min runs: 3" isn't in your source files and conflicts with the rule that fewer than 4 firing seeds is an automatic fail. | Dropped; N36 states the 4-seed rule. |
| 47 | 11 | `mean 12836.` | Wrong or unverified fact | Same truncated mean as page 7 (true value 12836.8). | Ḡ = 12836.8 (2.6). |
| 48 | 11 | `Spread in grok times lets test tracking.` | Style / filler | Garbled sentence. | "that spread is what lets you test whether a predictor tracks when grok happens". |
| 49 | 13 | `Fraction test correct. Only thing defining grok. 8100/8939 ≈ 0.906.` | Wrong or unverified fact | 8100 is a made-up count presented as if it were yours. | 3.1 Dass block derives the real threshold: at least 8,046 of 8,939 test pairs must be right; 8,100 kept only as a labelled example. |
| 50 | 13 | `Primary threshold first entry of grok thresholds=0.9.` | New term out of nowhere | Config jargon with no explanation. | N27: "the grok threshold is the scalar 0.9 (the first entry of grok_thresholds in the config)". |
| 51 | 13 | `G = (12987, 6921, 9510, 11019, 23747).` | Inconsistent symbol / type | G is a 5-entry list here but a single scalar in 4.4. | G_i is seed i's grok epoch; the list is written (G_0,...,G_4). |
| 52 | 15 | `L2 crossover P = (113, 113, 120, 131, 137) precisely 113.4,...` | Inconsistent symbol / type | Two versions of P; 4.2 and 4.4 then compute with 113, 5.1 with 113.4. | The book always uses 113.4, 113.2, 120.1, 130.6, 136.8; rounding appears only in the 5.2 tie example. |
| 53 | 15 | `precisely 113.4,113.2,120.1,130.6,136.8` | New term out of nowhere | Non-integer epochs are never explained. | N32 (log-epoch grid, interpolation) explains them. |
| 54 | 15 | `Firing counts as led only if P < G − 100.` | Used before defined | "firing" used before "fire" is defined (4.5 only says "never fires"). | N30 defines fire and miss. |
| 55 | 15 | `Summary table “before grok” uses > 0 not 100 margin` | New term out of nowhere | "summary table" undefined. | N54: the L2 results table of 26 Sep 2026. |
| 56 | 15 | `lead frac = #{ P < G − 100}/5` | Inconsistent symbol / type | Denominator 5 here, but 4.5 uses 4/4 (number of firing seeds): the definition changes between sections. | One definition: lead_frac = #{i in F : P_i < G_i − 100}/n with n = #F (N36, N38). |
| 57 | 15 | `#{ P < G − 100}` | Used before defined | # (count) never defined. | A1 defines sets, ∈ and #. |
| 58 | 15 | `mean r̄ over firing seeds` | Used before defined | "firing seeds" used in 4.4; the dropping rule comes only in 4.5. | F and n defined in N36. |
| 59 | 15 | `1. G = 12987 seed 0.  2. P = 113 seed 0.` | Inconsistent symbol / type | Uses the rounded P = 113 while 5.1 uses 113.4. | 4.4 uses P_0 = 113.4. |
| 60 | 16 | `= 0.9913 ∵ division ... r ≈ 0.991` | Dass-layout or equation-rule breach | = used for a rounded division, then a different rounding with ≈ in the box. | One ≈ line where digits are dropped; box reads r_0 = 0.9913 (4 d.p.). |
| 61 | 16 | `Mean 0.989 > 0.9 fails upper.` | Collapsed or missing step | 0.989 is never calculated. | 4.4 second Dass block computes r̄ = 0.988934 (0.989 to 3 d.p.). |
| 62 | 16 | `evaluate frozen protocol drops None before ρ and r̄` | Used before defined | Function name and ρ used before their sections (6.1, 5.1). | Function in N45; ρ in N42; the dropping rule in N36. |
| 63 | 16 | `If < 4 fire, immediate fail lead frac=0, ρ=NaN, mean 0.` | Wrong or unverified fact | The values 0, NaN, 0 aren't in your source files; NaN undefined. | Kept only "fewer than 4 firing seeds is an automatic FAIL" (N36, N45). |
| 64 | 17 | `−1 to 1 whether two lists go up together in order, using only ranks.` | Used before defined | Rank never defined. | N40 defines rank and ties. |
| 65 | 17 | `6 ∑ d2 / n(n2 − 1), di rank difference` | Used before defined | n undefined; d_i given only as "rank difference" without sign convention. | n = #F (N36); d_i = R(G_i) − R(P_i) (N42). |
| 66 | 17 | `3. Rank definition.` | Dass-layout or equation-rule breach | A Given line that names no fact or source. | Given 4: "Rank: position from smallest, starting at 1 (N40)". |
| 67 | 18 | `= 1 − 36/120  ∵ 6 × 6 = 36, 5 × 24 = 120` | Collapsed or missing step | Two multiplications in one line, and 25 − 1 = 24 skipped. | Separate lines for 5² − 1 = 24, 6 × 6 = 36, 5 × 24 = 120. |
| 68 | 18 | `If rounded to (113, 113, 120, 131, 137) tie at 1.5, ρ ≈ 0.56.` | Collapsed or missing step | No working; "tie" undefined; the method (Pearson on ranks) not given and Pearson not yet defined. | 5.2 Dass block: average ranks, means, deviations, three sums, ρ = 5.5/√95 = 0.564. |
| 69 | 18 | `Pearson uses raw values; Spearman only order.` | New term out of nowhere | Pearson's correlation is never defined. | N41 defines C(X, Y) with its formula. |
| 70 | 18 | `ρ = 1 − ∑ d2 /20, ∑ d2 even 0..40` | Collapsed or missing step | No reason given for /20, for "even", or for the maximum 40. | 5.3 Dass block derives all three, and 21 values. |
| 71 | 18 | `at most one adjacent swap` | New term out of nowhere | "adjacent swap" undefined. | N43. |
| 72 | 18 | `Code uses RHO THRESH just below 0.9` | New term out of nowhere | Constant name with no value. | N45 gives the coded value 0.8999999999999998 and why. |
| 73 | 18 | `only ρ = 1 (and borderline 0.9) unlikely by chance; ρ = 0.70 not significant` | New term out of nowhere | "by chance" and "significant" undefined. | N44 defines n!, chance, p-value, significant. |
| 74 | 18 | `Random ordering gives ρ = 1 prob 1/120.` | Collapsed or missing step | No working for 120 = 5!. | 5.4 Dass block: 5! = 120, Pr(ρ = 1) = 1/120, Pr(ρ ≥ 0.9) = 1/24. |
| 75 | 19 | `Written 29 Sep 2026, after L2 closed` | New term out of nowhere | "closed" undefined. | N47 closed-negative. |
| 76 | 19 | `Only PRIMARY SIGNALS receive verdict.` | New term out of nowhere | Code constant with no definition; not in your source files. | Dropped; 6.1 self-check now says which predictor was left out of scoring (L2, 2 Oct). |
| 77 | 19 | `Each blocks different failure: late, fixed clock, absurdly early.` | New term out of nowhere | "fixed clock" undefined. | N42 defines fixed clock. |
| 78 | 19 | `L2 crossover: 1.0 pass, 0.70 fail, 0.989 fail → FAIL.` | Dass-layout or equation-rule breach | Arrow used instead of a sentence or equation (also "→ PASS", "200→100"). | Plain words: "so FAIL". |
| 79 | 19 | `MA-of-MA: 1.0 pass, 0.00 fail, 0.835 pass` | New term out of nowhere | The MA-of-MA rule is never defined anywhere. | N33 defines it with its P values; 6.2 Dass block derives 1.0, 0, 0.835. |
| 80 | 19 | `Perfect P = 0.5G gives 1,1,0.5 → PASS.` | Collapsed or missing step | No working. | 6.3 Dass block. |
| 81 | 19 | `context.md records calibration. If done after some results known, post-hoc.` | New term out of nowhere | context.md, calibration and post-hoc undefined. | A9 and N48. |
| 82 | 19 | `fixed P = 1800 ρ = 0 must FAIL` | Wrong or unverified fact | With every P equal, all ranks tie and Pearson-on-ranks divides by zero, so ρ is undefined (NaN), not 0. It still fails. | 6.3 and 5.1 self-check say ρ is undefined and ask you to check what the code returns. |
| 83 | 19 | `Synthetic: perfect P = 0.5G must PASS` | New term out of nowhere | "synthetic" undefined. | N48. |
| 84 | 21 | `Two rules (L2 halfway down from peak, Dropout at 10% rise)` | New term out of nowhere | Neither rule is defined; the Dropout predictor appears for the first time here. | N31 lists the 8 predictors; N33 defines the halfway reading; N34 the Dropout 10% rule. |
| 85 | 21 | `Target = peak −0.5(peak − yfinal )` | Inconsistent symbol / type | "peak" in the text, y_peak in the derivation; y_final undefined at first use. | y_peak, y_final in N28; y_target in N33. |
| 86 | 21 | `1. y(t) scalar signal. ... 1. y: vector 40001 entries.` | Inconsistent symbol / type | The signal is a scalar in Given and a vector in Let; 40001 entries isn't in your source files. | y_t is a scalar at epoch t (N28) everywhere. |
| 87 | 21 | `To find:Show ytarget needs future. ... Uses future yfinal` | Dass-layout or equation-rule breach | Box holds a phrase, and the rearrangement doesn't show the dependence on the future. | 7.1 block: factoring step, ∴ sentence explaining the 0.5 share of y_final, boxed y_target = 0.5 y_peak + 0.5 y_final. |
| 88 | 21 | `uniform filter1d centred` | New term out of nowhere | Function name used without saying what it is. | N29. |
| 89 | 22 | `ȳcausal t ... yt scalar signal at t` | Inconsistent symbol / type | y_t here, y(t) in 7.1. | y_t throughout. |
| 90 | 22 | `Given: 1. yt scalar raw signal. Let: 1. yt : scalar signal at t.` | Dass-layout or equation-rule breach | Given and Let repeat the same line. | Given cites N28/N29; Let defines each average once. |
| 91 | 22 | `Example: L2 skip epochs 200→100` | New term out of nowhere | skip_epochs undefined; arrow notation. | N35; "went from 200 to 100". |
| 92 | 22 | `Which settings say “not from a paper”? All l2 norm windows.` | Wrong or unverified fact | The answer "all l2 norm windows" isn't in your source files. | Self-check now asks you to check each window and skip value for a citation. |
| 93 | 22 | `Near-miss L2 peak r̄ = 0.904` | New term out of nowhere | "L2 peak" event and "near-miss" undefined. | N33 peak event, N51 near-miss; 7.4 Dass block gives δ = 0.004. |
| 94 | 22 | `run with λ = 0 (often doesn’t grok) would show ∑ w2` | Inconsistent symbol / type | w again instead of θ. | S_t. |
| 95 | 22 | `Seven of eight predictors closed-negative` | New term out of nowhere | "closed-negative" undefined; the eight predictors are never listed. | N31, N47. |
| 96 | 23 | `Appendix A: Full Commutator Defect Derivation` | Wrong or unverified fact | Predictor-specific maths in the common-concepts book; the syllabus says commutators are left for later. | Removed from this book; corrections listed below so they go into the Commutator Defect book. |
| 97 | 23 | `We derive two SGD steps on batches A then B.` | Wrong or unverified fact | Plain SGD on two batches isn't your setup: your runs are full-batch AdamW. | For the Commutator Defect book: state which step the predictor code uses. |
| 98 | 23 | `6. O(η 2 ): scalar remainder` | Inconsistent symbol / type | The remainder in a vector equation is a vector, not a scalar. | For the Commutator Defect book: "vector with M entries whose size is at most a constant times η²". |
| 99 | 23 | `θk : vector n entries, weights at step k` | Inconsistent symbol / type | Step index k and n entries, while the rest of the book uses epochs and other symbols. | For the Commutator Defect book: use θ_t and M (N9, N10). |
| 100 | 23 | `∵ Given 4 definition one SGD ste / ∵ Taylor gradient, θk+1 − θk = O` | Layout | The reasons column runs off the right edge of the page and is cut off. | Reasons are wrapped in a fixed-width column in v2; no overfull lines. |
| 101 | 23 | `Taylor gradient` | New term out of nowhere | Taylor expansion never defined. | For the Commutator Defect book: define it in Let. |
| 102 | 23 | `∵ η scalar pull out ∵ η scalar` | Layout | Doubled "because". | Not applicable in v2 (appendix removed). |
| 103 | 23 | `∵ distribute − η ∵ (−)(−) = +` | Collapsed or missing step | Distributing and the sign change in one line. | For the Commutator Defect book: split into two lines. |
| 104 | 23 | `= ... −ηO(η 2 ) ∵ ηO(η 2 ) = O(η 2 ) ∵ absorption` | Wrong or unverified fact | η·O(η²) is O(η³). Keeping the remainder as O(η²) swallows the η² H_B g_A term the result claims to keep. | For the Commutator Defect book: remainder O(η³), so θ^{AB} = θ − η(g_A + g_B) + η² H_B g_A + O(η³). |
| 105 | 23 | `≈ θk − η ( g A (θk ) + gB (θk ))` | Wrong or unverified fact | The ≈ line drops the η² term, but the boxed answer on page 24 keeps it: the two contradict each other. | For the Commutator Defect book: no ≈ line; box the O(η³) form. |
| 106 | 24 | `CHAPTER 7. MODULE 7: RESEARCH HONESTY (header on Appendix page)` | Layout | Appendix pages carry the Module 7 running header. | No running headers in v2 (style guide: page number only). |
| 107 | 24 | `This follows all 10 rules.` | Style / filler | Filler; the "10 rules" are never stated. | Removed. |
| 108 | 25 | `1. Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting.` | Wrong or unverified fact | Title cut short (missing "on Small Algorithmic Datasets"); refs 3 and 4 lack title or venue details. | Full references as in the syllabus material. |
| 109 | 7 | `(whole body) e.g. "Share firing seeds that lead." "Fraction test correct."` | Style / filler | Telegraphic fragments instead of plain short sentences. | Full short sentences addressed to you. |

## Problems in the source material itself (not visible in the PDF)

These were in the syllabus material or the facts list. v2 follows the frozen-protocol facts in CHAT_SUMMARY.

| Where | Quote | Problem and what v2 does |
|---|---|---|
| Syllabus material 4.5 | "With one miss and four leads, lead_frac = 0.8" | Conflicts with the dropping rule in CHAT_SUMMARY (misses dropped, so 4/4 = 1.0). v2 uses 1.0 and flags it as an attack point. |
| Syllabus material 4.3 self-check | "one seed misses and one fires after grok ... 3/5 = 0.6" | Under the dropping rule it is 3/4 = 0.75 (still a fail). v2 uses 3/4. |
| Syllabus material 4.2 | "A miss counts as no lead." | Same conflict; v2 follows the dropping rule. |
| Syllabus material 2.5 | "If checkpoints are saved every 100 epochs" | Your facts say saved every 50, evaluated every 100. v2 uses 50/100. |
| Syllabus material 2.6 | "A seed fixes ... batch order" | Your runs are full batch (L2 material), so there is no batch order. v2 says initial weights and data split. |
| Syllabus material 6.3 | "a fixed P = 1800 gives ρ = 0" | ρ is undefined for a constant list (see main table). |
| Syllabus header / CHAT_SUMMARY | "7 modules, 31 topics" | The syllabus actually lists 29 topics (3+6+2+5+4+3+6), and so does the original PDF. v2 has all 29 and says 29. |
| Your facts vs G values | checkpoints evaluated every 100 epochs; G = 12987 etc. | G values are not multiples of 50 or 100, so G must come from a finer accuracy log. Worth confirming in analyze_nanda_unified.py. |

## Scope decisions in v2

- Appendix A (Commutator Defect) is not carried over: it is predictor-specific, uses a plain-SGD two-batch step that isn't your full-batch AdamW setup, and its maths had the errors listed above. The corrected form for the Commutator Defect book is θ^{AB}_{t+2} = θ_t − η(g_A + g_B) + η² H_B g_A + O(η³).
- Unverifiable items were dropped rather than guessed: vocab = 114, `min runs: 3`, `PRIMARY_SIGNALS`, the NaN/0 values for fewer than 4 firing seeds, `y` with 40001 entries, and "all l2 norm windows".
- The learning rate η and the weight count M are named but given no values, because your source files don't give them.
