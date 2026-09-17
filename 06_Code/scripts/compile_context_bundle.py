#!/usr/bin/env python3
"""
compile_context_bundle.py
==========================

Builds `full_context_profile.pdf` in the repo root — a curated, single-file
context bundle meant to be handed to a NEW chat session (Meta AI or any
other assistant) that has no memory of this project.

This is NOT a "compile everything" tool like the other tools/compile_*.py
scripts. It deliberately picks the smallest set of files/sections that let a
fresh session pick up the thesis work correctly: project identity, the last
two session summaries from context.md, the benchmark config, latest results,
the master runner's key parts, the two CLOSED predictors' full code, and the
shared model/data files.

Usage:
    python tools/compile_context_bundle.py
"""

import json
import os
import subprocess
import sys
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PDF = os.path.join(REPO_ROOT, "full_context_profile.pdf")

TODAY = date.today().isoformat()

EXCLUDED_NOTES = [
    ("08_Experiments/results/nanda_unified/ (legacy: was runs/four_head/)", "old 3-run pipeline, superseded by results/nanda_unified/ "
                         "(run_full_benchmark.py itself is deprecated/archived)"),
    ("archive/", "historical pre-Nanda-Unified data, not needed to continue current work"),
    ("08_Experiments/results/nanda_unified/seed_*/checkpoints/*.pt", "large binary model weights, "
     "not readable context, and not needed to explain the benchmark"),
    ("08_Experiments/results/nanda_unified/seed_*/{l2_norm,dropout}/*.npy", "large raw numeric arrays; "
     "the per-seed summary.json already carries the derived predictor signals"),
    ("graphify-out/", "generated knowledge-graph cache, rebuildable from source anytime"),
    ("Literature/*.pdf", "source papers themselves - cited by name in context.md, "
     "not needed verbatim for a new chat to continue engineering work"),
    ("context.md sessions before the last 2", "older history superseded by the L2-Norm "
     "and Dropout closure decisions already summarized in the last 2 sessions"),
]


def run_git(args):
    try:
        return subprocess.run(
            ["git"] + args, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:
        return f"(git command failed: {exc})"


def read_text(path, max_lines=None):
    full_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    truncated = False
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    text = "".join(lines)
    if truncated:
        text += "\n... [truncated] ...\n"
    return text


def read_text_head_tail(path, head=350, tail=50):
    full_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    if len(lines) <= head + tail:
        return "".join(lines)
    gap = len(lines) - head - tail
    return (
        "".join(lines[:head])
        + f"\n... [gap: {gap} lines omitted] ...\n\n"
        + "".join(lines[-tail:])
    )


def extract_last_n_sessions(context_md_text, n=2):
    """Split context.md on '## Session Summary' / '### Session Summary'
    headers and return the last n session blocks, full text."""
    lines = context_md_text.splitlines()
    header_idx = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("## Session Summary")
        or line.strip().startswith("### Session Summary")
    ]
    if not header_idx:
        return context_md_text[-8000:]
    chosen_starts = header_idx[-n:]
    start = chosen_starts[0]
    return "\n".join(lines[start:])


def extract_runner_constants_and_registry(runner_text, max_lines=400):
    """Pull out the constants block, CHECKPOINT_PREDICTOR_FUNCS registry, and
    the _checkpoint_predictor_dropout_variance docstring from
    run_nanda_benchmark.py, capped at max_lines total."""
    lines = runner_text.splitlines()
    chunks = []

    def grab(start_marker, end_marker=None, max_span=200):
        for i, line in enumerate(lines):
            if start_marker in line:
                j = i
                count = 0
                block = []
                while j < len(lines) and count < max_span:
                    block.append(lines[j])
                    if end_marker and end_marker in lines[j] and j > i:
                        break
                    j += 1
                    count += 1
                return "\n".join(block)
        return None

    constants = grab("GROK_ACC_THRESHOLD = ", end_marker="DROPOUT_VARIANCE_NUM_CHECKPOINTS", max_span=40)
    if constants:
        chunks.append("--- Constants block ---\n" + constants)

    docstring = grab('def _checkpoint_predictor_dropout_variance', end_marker='"""', max_span=25)
    if docstring:
        # grab up to the closing triple-quote (second occurrence)
        for i, line in enumerate(lines):
            if 'def _checkpoint_predictor_dropout_variance' in line:
                block = []
                quote_count = 0
                j = i
                while j < len(lines) and quote_count < 2:
                    block.append(lines[j])
                    quote_count += lines[j].count('"""')
                    j += 1
                docstring = "\n".join(block)
                break
        chunks.append("\n--- _checkpoint_predictor_dropout_variance docstring "
                       "(reference shape for future checkpoint-only predictors "
                       "like Spectral) ---\n" + docstring)

    registry = grab("CHECKPOINT_PREDICTOR_FUNCS = {", end_marker="CHECKPOINT_ONLY_PREDICTORS = ", max_span=15)
    if registry:
        chunks.append("\n--- CHECKPOINT_PREDICTOR_FUNCS registry ---\n" + registry)

    text = "\n".join(chunks)
    text_lines = text.splitlines()
    if len(text_lines) > max_lines:
        text = "\n".join(text_lines[:max_lines]) + "\n... truncated ..."
    return text


def build_seed_table():
    rows = [("seed", "grok_epoch", "l2_ma_crossover_epoch", "dropout_variance_peak_epoch", "dropout_gap@0.5")]
    seed_dir = os.path.join(REPO_ROOT, "results", "nanda_unified")
    if not os.path.isdir(seed_dir):
        return rows
    for name in sorted(os.listdir(seed_dir)):
        if not name.startswith("seed_"):
            continue
        summary_path = os.path.join(seed_dir, name, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        grok_epoch = summary.get("grok_epoch", "-")
        l2_epoch = summary.get("l2_predictor", {}).get("ma_crossover_epoch", "-")
        dv_epoch = summary.get("dropout_variance_predictor", {}).get("variance_peak_epoch", "-")
        gap05 = summary.get("dropout_final_gap_by_rate", {}).get("0.5", "-")
        rows.append((
            name,
            str(grok_epoch),
            f"{l2_epoch:.1f}" if isinstance(l2_epoch, float) else str(l2_epoch),
            str(dv_epoch),
            f"{gap05:.4f}" if isinstance(gap05, float) else str(gap05),
        ))
    return rows


def pretty_json(path, max_lines=300):
    full_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_path):
        return f"(not found: {path})"
    with open(full_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    text = json.dumps(data, indent=2)
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n... [truncated] ..."
    return text


def build_with_reportlab():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak,
    )
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBig", parent=styles["Title"], fontSize=20, spaceAfter=14)
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=15, spaceBefore=6, spaceAfter=8,
        textColor=colors.HexColor("#1a3d6d"))
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle(
        "BodyBig", parent=styles["BodyText"], fontSize=10, leading=13, alignment=TA_LEFT)
    mono = ParagraphStyle(
        "Mono", fontName="Courier", fontSize=7.2, leading=8.6)
    filehdr = ParagraphStyle(
        "FileHdr", fontName="Courier-Bold", fontSize=9, spaceBefore=8, spaceAfter=3,
        textColor=colors.HexColor("#7a1f1f"))

    current_section = {"name": ""}

    def footer(canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        text = f"Context Bundle - for new chat - {current_section['name']} - {TODAY}"
        canvas_obj.drawString(0.6 * inch, 0.4 * inch, text)
        canvas_obj.drawRightString(A4[0] - 0.6 * inch, 0.4 * inch, f"p.{doc.page}")
        canvas_obj.restoreState()

    def set_section(name):
        current_section["name"] = name

    def esc(text):
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    story = []

    # ---- Page 1: Title ----
    set_section("title")
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "Thesis Full Context Profile<br/>Unified Benchmark of Grokking Predictors",
        title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Generated: {TODAY}", body))
    story.append(Paragraph("Generated for new chat session", body))
    commit_hash = run_git(["rev-parse", "--short", "HEAD"])
    story.append(Paragraph(f"Git commit: {esc(commit_hash)}", body))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Last 10 commits:", h2))
    log = run_git(["log", "--oneline", "-10"])
    story.append(Preformatted(esc(log), mono))
    story.append(PageBreak())

    # ---- Section 1: Project Identity ----
    set_section("context.md (project identity)")
    story.append(Paragraph("Section 1 - Project Identity", h1))
    ctx_text = read_text("context.md") or ""
    top120 = "\n".join(ctx_text.splitlines()[:120])
    story.append(Preformatted(esc(top120), mono))
    story.append(PageBreak())

    # ---- Section 2: Current Progress Snapshot ----
    set_section("context.md (last 2 sessions)")
    story.append(Paragraph("Section 2 - Current Progress Snapshot (last 2 sessions)", h1))
    story.append(Paragraph(
        "<b>HIGHLIGHTS:</b> Predictor 1 (L2-Norm) = CLOSED, NEGATIVE. "
        "Predictor 2 (Dropout) = CLOSED, NEGATIVE (k=30, rate=0.5 deviation "
        "from Salah &amp; Yevick justified on compute grounds). "
        "Predictor 3 (Spectral) = CLOSED, NEGATIVE (k_90 3448-&gt;~3350-3400, "
        "no rank collapse; Canatar task-model alignment). "
        "Predictor 4 (AGE) = CLOSED, NEGATIVE (NC1 collapses ~45-&gt;~0.05 but "
        "after grok in 5/5 seeds). "
        "Predictor 5 (HTSR Alpha) = NEXT, not started.",
        body))
    story.append(Spacer(1, 0.1 * inch))
    last2 = extract_last_n_sessions(ctx_text, n=2)
    for chunk_start in range(0, len(last2), 6000):
        story.append(Preformatted(esc(last2[chunk_start:chunk_start + 6000]), mono))
    story.append(PageBreak())

    # ---- Section 3: Benchmark Config ----
    set_section("06_Code/configs/nanda_unified.yaml")
    story.append(Paragraph("Section 3 - Benchmark Config", h1))
    story.append(Paragraph("=== configs/nanda_unified.yaml ===", filehdr))
    cfg_text = read_text("06_Code/configs/nanda_unified.yaml") or "(not found)"
    story.append(Preformatted(esc(cfg_text), mono))
    story.append(PageBreak())

    # ---- Section 4: Latest Results ----
    set_section("08_Experiments/results/nanda_unified")
    story.append(Paragraph("Section 4 - Latest Results", h1))
    story.append(Paragraph("=== results/nanda_unified/aggregate.json ===", filehdr))
    story.append(Preformatted(esc(pretty_json("08_Experiments/results/nanda_unified/aggregate.json")), mono))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Per-seed summary extract:", h2))
    rows = build_seed_table()
    row_text = "\n".join(
        f"{r[0]:<10} {r[1]:<12} {r[2]:<24} {r[3]:<28} {r[4]:<12}" for r in rows)
    story.append(Preformatted(esc(row_text), mono))
    story.append(PageBreak())

    # ---- Section 5: Master Runner Key Parts ----
    set_section("run_nanda_benchmark.py")
    story.append(Paragraph("Section 5 - Master Runner Key Parts", h1))
    story.append(Paragraph("=== run_nanda_benchmark.py (extract) ===", filehdr))
    runner_text = read_text("run_nanda_benchmark.py") or ""
    extract = extract_runner_constants_and_registry(runner_text, max_lines=400)
    story.append(Preformatted(esc(extract), mono))
    story.append(PageBreak())

    # ---- Section 6: Predictor Implementations ----
    set_section("src/predictors")
    story.append(Paragraph("Section 6 - Predictor Implementations (closed predictors)", h1))
    for pfile in ("src/predictors/l2_norm.py", "src/predictors/dropout.py"):
        story.append(Paragraph(f"=== {pfile} ===", filehdr))
        text = read_text(pfile) or "(not found)"
        story.append(Preformatted(esc(text), mono))
        story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("=== src/predictors/spectral.py ===", filehdr))
    spectral_text = read_text("src/predictors/spectral.py")
    if spectral_text is None:
        story.append(Paragraph("Spectral not yet implemented - stub expected", body))
    else:
        story.append(Preformatted(esc(spectral_text), mono))
    story.append(PageBreak())

    # ---- Section 7: Measurements & Model ----
    set_section("measurements/model")
    story.append(Paragraph("Section 7 - Measurements & Model", h1))
    story.append(Paragraph("=== src/unified_measurements.py (first 200 lines) ===", filehdr))
    story.append(Preformatted(esc(read_text("src/unified_measurements.py", max_lines=200) or ""), mono))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("=== src/models/transformer_four_head.py (first 150 lines) ===", filehdr))
    story.append(Preformatted(esc(read_text("src/models/transformer_four_head.py", max_lines=150) or ""), mono))
    story.append(PageBreak())

    # ---- Section 8: Data Pipeline ----
    set_section("src/data/modular_arithmetic.py")
    story.append(Paragraph("Section 8 - Data Pipeline", h1))
    story.append(Paragraph("=== src/data/modular_arithmetic.py ===", filehdr))
    story.append(Preformatted(esc(read_text("src/data/modular_arithmetic.py") or ""), mono))
    story.append(PageBreak())

    # ---- Section 9: Predictor Evaluation Order & Open Questions ----
    set_section("predictor order / open questions")
    story.append(Paragraph("Section 9 - Predictor Evaluation Order & Open Questions", h1))
    order = [
        ("1. L2 Norm", "CLOSED - negative"),
        ("2. Dropout", "CLOSED - negative (k=30, rate=0.5 deviation justified)"),
        ("3. Spectral", "CLOSED - negative (v4; k_90 3448->~3350-3400, no collapse)"),
        ("4. AGE", "CLOSED - negative (v4; NC1 collapses after grok in 5/5)"),
        ("5. HTSR Alpha", "NEXT - not started"),
        ("6. Correlation Traps", "not started"),
        ("7. Weight-PCA", "not started"),
        ("8. Higher-MI", "not started"),
        ("9. Commutator Defect", "not started"),
    ]
    order_text = "\n".join(f"{name:<24} {status}" for name, status in order)
    story.append(Preformatted(esc(order_text), mono))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Predictors 1-4 are all CLOSED negative under the same two-criterion "
        "falsification test (signal extremum does not lead grok in 5/5 seeds; "
        "failure direction not seed-consistent). Next open work item: "
        "Predictor 5, HTSR Alpha (Martin &amp; Mahoney weightwatcher heavy-tailed "
        "self-regularisation, per-weight-matrix ESD power-law exponent alpha) - "
        "checkpoint-only plugin pattern, not yet started.", body))

    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=A4,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Thesis Full Context Profile",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build_with_matplotlib_fallback():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    sections = []

    commit_hash = run_git(["rev-parse", "--short", "HEAD"])
    log = run_git(["log", "--oneline", "-10"])
    sections.append(("title", "Thesis Full Context Profile - Unified Benchmark of "
                      f"Grokking Predictors\nGenerated: {TODAY}\n"
                      "Generated for new chat session\n"
                      f"Git commit: {commit_hash}\n\nLast 10 commits:\n{log}"))

    ctx_text = read_text("context.md") or ""
    top120 = "\n".join(ctx_text.splitlines()[:120])
    sections.append(("context.md (identity)", "SECTION 1 - PROJECT IDENTITY\n\n" + top120))

    last2 = extract_last_n_sessions(ctx_text, n=2)
    sections.append(("context.md (last 2 sessions)",
                      "SECTION 2 - CURRENT PROGRESS SNAPSHOT\n\n"
                      "HIGHLIGHTS: Predictor 1 (L2-Norm) = CLOSED, NEGATIVE. "
                      "Predictor 2 (Dropout) = CLOSED, NEGATIVE. "
                      "Predictor 3 (Spectral) = CLOSED, NEGATIVE (k_90 3448->3500, "
                      "no collapse). Predictor 4 (AGE) = CLOSED, NEGATIVE (NC1 "
                      "collapses after grok). Predictor 5 (HTSR Alpha) = NEXT.\n\n"
                      + last2))

    sections.append(("06_Code/configs/nanda_unified.yaml",
                      "SECTION 3 - BENCHMARK CONFIG\n\n=== configs/nanda_unified.yaml ===\n"
                      + (read_text("06_Code/configs/nanda_unified.yaml") or "")))

    rows = build_seed_table()
    row_text = "\n".join(
        f"{r[0]:<10} {r[1]:<12} {r[2]:<24} {r[3]:<28} {r[4]:<12}" for r in rows)
    sections.append(("08_Experiments/results/nanda_unified",
                      "SECTION 4 - LATEST RESULTS\n\n=== aggregate.json ===\n"
                      + pretty_json("08_Experiments/results/nanda_unified/aggregate.json")
                      + "\n\nPer-seed summary extract:\n" + row_text))

    runner_text = read_text("run_nanda_benchmark.py") or ""
    extract = extract_runner_constants_and_registry(runner_text, max_lines=400)
    sections.append(("run_nanda_benchmark.py",
                      "SECTION 5 - MASTER RUNNER KEY PARTS\n\n" + extract))

    pred_text = "SECTION 6 - PREDICTOR IMPLEMENTATIONS\n\n"
    for pfile in ("src/predictors/l2_norm.py", "src/predictors/dropout.py"):
        pred_text += f"=== {pfile} ===\n" + (read_text(pfile) or "(not found)") + "\n\n"
    spectral_text = read_text("src/predictors/spectral.py")
    pred_text += "=== src/predictors/spectral.py ===\n"
    pred_text += spectral_text if spectral_text is not None else "Spectral not yet implemented - stub expected"
    sections.append(("src/predictors", pred_text))

    meas_text = ("SECTION 7 - MEASUREMENTS & MODEL\n\n"
                 "=== src/unified_measurements.py (first 200 lines) ===\n"
                 + (read_text("src/unified_measurements.py", max_lines=200) or "")
                 + "\n\n=== src/models/transformer_four_head.py (first 150 lines) ===\n"
                 + (read_text("src/models/transformer_four_head.py", max_lines=150) or ""))
    sections.append(("measurements/model", meas_text))

    sections.append(("src/data/modular_arithmetic.py",
                      "SECTION 8 - DATA PIPELINE\n\n=== src/data/modular_arithmetic.py ===\n"
                      + (read_text("src/data/modular_arithmetic.py") or "")))

    order = [
        ("1. L2 Norm", "CLOSED - negative"),
        ("2. Dropout", "CLOSED - negative (k=30, rate=0.5 deviation justified)"),
        ("3. Spectral", "CLOSED - negative (v4; k_90 3448->~3350-3400, no collapse)"),
        ("4. AGE", "CLOSED - negative (v4; NC1 collapses after grok in 5/5)"),
        ("5. HTSR Alpha", "NEXT - not started"),
        ("6. Correlation Traps", "not started"),
        ("7. Weight-PCA", "not started"),
        ("8. Higher-MI", "not started"),
        ("9. Commutator Defect", "not started"),
    ]
    order_text = "\n".join(f"{name:<24} {status}" for name, status in order)
    sections.append(("predictor order",
                      "SECTION 9 - PREDICTOR EVALUATION ORDER & OPEN QUESTIONS\n\n"
                      + order_text
                      + "\n\nPredictors 1-4 all CLOSED negative under the same "
                        "two-criterion falsification test. Next open work item: "
                        "Predictor 5, HTSR Alpha (checkpoint-only plugin pattern), "
                        "not yet started."))

    with PdfPages(OUTPUT_PDF) as pdf:
        for name, text in sections:
            lines = text.splitlines()
            lines_per_page = 80
            for start in range(0, max(1, len(lines)), lines_per_page):
                chunk = "\n".join(lines[start:start + lines_per_page])
                fig = plt.figure(figsize=(8.27, 11.69))  # A4
                fig.text(0.05, 0.97, chunk, family="monospace", fontsize=6,
                         va="top", ha="left", wrap=True)
                fig.text(0.05, 0.02, f"Context Bundle - for new chat - {name} - {TODAY}",
                         fontsize=6)
                pdf.savefig(fig)
                plt.close(fig)


def main():
    try:
        build_with_reportlab()
        engine = "reportlab"
    except ImportError:
        build_with_matplotlib_fallback()
        engine = "matplotlib fallback"

    size_kb = os.path.getsize(OUTPUT_PDF) / 1024.0

    page_count = "?"
    try:
        from pypdf import PdfReader
        page_count = len(PdfReader(OUTPUT_PDF).pages)
    except Exception:
        try:
            import subprocess as sp
            out = sp.run(["mdls", "-name", "kMDItemNumberOfPages", OUTPUT_PDF],
                         capture_output=True, text=True).stdout
            page_count = out.strip().split("=")[-1].strip()
        except Exception:
            pass

    print(f"Saved: full_context_profile.pdf ({page_count} pages, {size_kb:.1f} KB) [{engine}]")
    print("Includes: context.md (2 latest sessions), aggregate.json, 5 seed summaries, "
          "nanda_unified.yaml, 2 predictors (l2_norm, dropout) + spectral if present, "
          "model, measurements")
    print("Excluded (deliberately, with reason):")
    for path, reason in EXCLUDED_NOTES:
        print(f"  - {path}: {reason}")


if __name__ == "__main__":
    sys.exit(main())
