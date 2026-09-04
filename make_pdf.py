import json
import os

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

# Where the full 5-seed Nanda-Unified re-baseline results live. Anchored to
# this script's own directory (repo root), NOT the current working
# directory — os.walk(".") below still assumes cwd == repo root, but this
# path must not silently miss the results just because make_pdf.py was run
# from somewhere else (e.g. an IDE "run" button using the file's own
# folder as cwd). Not present until that benchmark has actually been run
# (results/ is gitignored, disk only).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON_PATH = os.path.join(REPO_ROOT, "results", "nanda_unified", "aggregate.json")
# Display-only label — always "results/nanda_unified/aggregate.json"
# regardless of cwd, since it's relative to REPO_ROOT not cwd.
RESULTS_JSON_LABEL = os.path.relpath(RESULTS_JSON_PATH, REPO_ROOT)


def _escape_line(line):
    """Escape a plain-text line for ReportLab's Paragraph XML/HTML parser,
    convert tabs, and preserve leading spaces via &nbsp;."""
    escaped = (
        line.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
        .rstrip("\n")
    )
    return escaped.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")


def add_text_lines(story, lines, style):
    """Append plain-text lines to the story as Paragraphs (or blank
    Spacers), using the same escaping as the source-code dump."""
    for line in lines:
        escaped_line = _escape_line(line)
        if not escaped_line.strip():
            story.append(Spacer(1, 4))
        else:
            story.append(Paragraph(escaped_line, style))


def _fmt_epoch(value):
    return "None" if value is None else f"{value:.2f}"


def _seed_result_lines(seed):
    """Format one seed's summary dict (matching results/nanda_unified/
    seed_*/summary.json / aggregate.json's "seeds" entries) into plain-text
    lines, mirroring the console table in run_nanda_benchmark.py's
    aggregate()."""
    lp = seed["l2_predictor"]
    gaps = seed["dropout_final_gap_by_rate"]
    lc = seed["limit_cycle_check"]
    w = lp["windows"]

    lines = [
        f"seed {seed['seed']}  (modulus={seed['modulus']}  epochs={seed['epochs']}  "
        f"wall_time={seed['wall_time_sec']}s)",
        f"  grok_epoch                    : {seed['grok_epoch']}",
        f"  final_train_acc               : {seed['final_train_acc']:.4f}",
        f"  final_test_acc                : {seed['final_test_acc']:.4f}",
        f"  l2_norm   init -> final        : "
        f"{seed['l2_norm_init']:.4f} -> {seed['l2_norm_final']:.4f}",
        f"  sum_w2    init -> final        : "
        f"{seed['sum_w2_init']:.3f} -> {seed['sum_w2_final']:.3f}",
        f"  token_embedding_share (init)  : {seed['token_embedding_share_init']:.4f}",
        "  L2 predictor:",
        f"    MA-crossover epoch           : {_fmt_epoch(lp['ma_crossover_epoch'])}",
        f"    MA-of-MA zero-crossing epoch : {_fmt_epoch(lp['ma_of_ma_zero_crossing_epoch'])}",
        f"    noise_floor                  : {lp['noise_floor']:.6f}",
        f"    windows: fast={w['fast']} slow={w['slow']} "
        f"ma_of_ma_fast={w['ma_of_ma_fast']} skip_epochs={w['skip_epochs']} "
        f"quiet_epoch_cutoff={w['quiet_epoch_cutoff']}",
        "  Dropout gap by rate:",
        "    " + "  ".join(f"p{r}={gaps[r]:+.4f}" for r in gaps),
    ]
    if lc.get("applicable"):
        label = "LIMIT CYCLE" if lc["limit_cycle"] else "stable"
        lines.append(
            f"  Limit-cycle check             : {label}  "
            f"window_start={lc['window_start_epoch']}  "
            f"post_grok_min={lc['post_grok_min']:.4f}  "
            f"post_grok_std={lc['post_grok_std']:.4f}  "
            f"post_grok_final={lc['post_grok_final']:.4f}  "
            f"dips<0.9={lc['epochs_below_0.9_post_grok']}"
        )
    else:
        lines.append(f"  Limit-cycle check             : n/a ({lc.get('reason', 'n/a')})")
    return lines


def add_results_section(story, header_style, results_style):
    """Append a RESULTS section built from results/nanda_unified/
    aggregate.json (the full 5-seed Nanda-Unified benchmark output). Skips
    cleanly with a note if the file does not exist yet (benchmark not run,
    or results/ not present on this machine)."""
    story.append(PageBreak())
    story.append(Paragraph(f"=== RESULTS: {RESULTS_JSON_LABEL} ===", header_style))
    story.append(Spacer(1, 5))

    if not os.path.isfile(RESULTS_JSON_PATH):
        print(f"No results found at {RESULTS_JSON_PATH} - skipping results section body.")
        add_text_lines(
            story,
            [f"(not found - run run_nanda_benchmark.py first to produce {RESULTS_JSON_LABEL})"],
            results_style,
        )
        return

    print(f"Processing results: {RESULTS_JSON_PATH}")
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as handle:
        agg = json.load(handle)

    summary_lines = [
        f"n_seeds          : {agg['n_seeds']}",
        f"epochs           : {agg['epochs']}",
        f"grok_epoch_mean  : {agg['grok_epoch_mean']}",
        f"grok_epoch_std   : {agg['grok_epoch_std']}",
        f"grok_epochs      : {agg['grok_epochs']}",
        f"n_limit_cycle    : {agg['n_limit_cycle']} / {agg['n_seeds']}",
    ]
    add_text_lines(story, summary_lines, results_style)
    story.append(Spacer(1, 10))

    for seed in agg["seeds"]:
        add_text_lines(story, _seed_result_lines(seed), results_style)
        story.append(Spacer(1, 10))


def compile_py_to_pdf(output_filename="combined_code.pdf"):
    """
    Find all Python (.py) files in the current project directory,
    excluding virtual environments and cache/build directories,
    compile their source code into a single PDF, then append the full
    Nanda-Unified benchmark results (results/nanda_unified/aggregate.json,
    complete per-seed numbers) as a final RESULTS section.
    """

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Style used for Python source code
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        spaceAfter=0,
        spaceBefore=0,
    )

    # Style used for file names / section headers
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading2"],
        fontName="Courier-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=15,
        spaceAfter=5,
    )

    # Style used for the results section (same monospace as code, slightly
    # larger so the numbers are easy to read on their own).
    results_style = ParagraphStyle(
        "ResultsStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        spaceAfter=0,
        spaceBefore=0,
    )

    story = []

    # Directories that should not be searched
    excluded_dirs = {
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".git",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
    }

    # Walk through the project directory
    for root, dirs, files in os.walk("."):

        # Prevent os.walk from entering excluded directories
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in excluded_dirs
        ]

        # Sort files for predictable PDF ordering
        for filename in sorted(files):

            # Only process Python files
            if not filename.endswith(".py"):
                continue

            # Do not include this PDF-generation script itself
            if filename == "make_pdf.py":
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath)

            print(f"Processing: {relative_path}")

            # Add file header
            story.append(
                Paragraph(
                    f"=== {relative_path} ===",
                    header_style,
                )
            )

            story.append(Spacer(1, 5))

            # Read the Python file
            try:
                with open(
                    filepath,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as file:
                    lines = file.readlines()

            except Exception as error:
                print(f"Skipping {relative_path}: {error}")
                continue

            # Add every line of source code
            add_text_lines(story, lines, code_style)

            # Space between Python files
            story.append(Spacer(1, 15))

    # Append the full benchmark results (complete numbers) as a final section
    add_results_section(story, header_style, results_style)

    # Generate the PDF
    doc.build(story)

    print()
    print(f"🎉 Successfully created: {output_filename}")


if __name__ == "__main__":
    compile_py_to_pdf()
