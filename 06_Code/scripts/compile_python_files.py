"""
compile_python_files.py
=======================

Compiles context.md, EVERY codeable file under the project root, all
results (08_Experiments/results), and the synthetic-control validation
results (controls_results) into one single PDF (`project_compilation.pdf`,
written in the project root).

Alongside the PDF, it generates a full Markdown data dump
(`project_compilation_npy_dump.md`) containing the complete, unabbreviated
contents of every .npy array in the results.

The project root is found from this file's location
(06_Code/scripts/ -> two levels up), so the script works on any machine.

Usage:
    python 06_Code/scripts/compile_python_files.py
"""

import json
import re
import textwrap
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "project_compilation.pdf"
OUTPUT_NPY_FILE = ROOT / "project_compilation_npy_dump.md"
CONTEXT_FILE = ROOT / "context.md"  # project memory, included at the start of the PDF

# Codeable = source code, scripts and code-like config. Data, images,
# weights (.npy/.png/.pt), papers (.pdf) and prose (.md) are not code.
CODE_EXTENSIONS = {
    ".py", ".gs", ".js", ".ts", ".sh", ".html", ".css",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".ipynb",
}
# Code files with no useful suffix, or a suffix added on top (e.g. *.py.deprecated).
CODE_NAMES = {"requirements.txt"}
CODE_SUFFIX_PAIRS = {".deprecated"}

SKIP_DIRS = {
    "__pycache__", ".git", ".idea", ".vs", ".vscode", ".claude",
    ".venv", "venv", "env", "build", "dist", "node_modules",
    ".pytest_cache", ".mypy_cache", "graphify-out",
}

# Results: text files go in as text, plots as images, .npy arrays as a summary
# (shape, dtype, stats, preview). Model checkpoints (*.pt, ~3.2 GB) are skipped.
RESULTS_DIR = ROOT / "08_Experiments" / "results"
# Synthetic-control validation experiment (run_control_experiment.py) writes here.
CONTROLS_DIR = ROOT / "controls_results"
RESULT_TEXT_EXT = {".json", ".log", ".md", ".csv"}
RESULT_EXTENSIONS = RESULT_TEXT_EXT | {".npy", ".png"}
NPY_FULL_LIMIT = 40   # arrays up to this many elements are printed in full
NPY_PREVIEW = 6      # otherwise: first and last N values

WRAP_WIDTH = 112
FONT_SIZE = 7


def should_skip(path: Path):
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def is_code_file(path: Path):
    if path in (OUTPUT_FILE, OUTPUT_NPY_FILE):
        return False
    if path.name in CODE_NAMES:
        return True
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return False
    if suffixes[-1] in CODE_SUFFIX_PAIRS and len(suffixes) > 1:
        return suffixes[-2] in CODE_EXTENSIONS
    return suffixes[-1] in CODE_EXTENSIONS


def collect_files():
    return sorted(
        f for f in ROOT.rglob("*")
        if f.is_file() and not should_skip(f) and is_code_file(f)
    )


def collect_results(base_dir: Path):
    if not base_dir.is_dir():
        return []
    return sorted(
        f for f in base_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in RESULT_EXTENSIONS
    )


def compact_json(text: str):
    """Put each innermost list on one line (content unchanged, far fewer pages)."""
    return re.sub(
        r"\[([^\[\]{}]*)\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)).strip() + "]",
        text,
    )


def summarize_npy(path: Path):
    import numpy as np

    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as exc:  # unreadable array: say so, keep going
        return f"(could not load: {exc})"
    lines = [f"shape: {arr.shape}   dtype: {arr.dtype}   elements: {arr.size}"]
    if arr.size and arr.dtype.kind in "biuf":
        lines.append(f"min: {arr.min():.6g}   max: {arr.max():.6g}   mean: {arr.mean():.6g}")
    flat = arr.reshape(-1)
    if arr.size <= NPY_FULL_LIMIT:
        lines.append(f"values: {flat.tolist()}")
    else:
        lines.append(f"first {NPY_PREVIEW}: {flat[:NPY_PREVIEW].tolist()}")
        lines.append(f"last {NPY_PREVIEW}: {flat[-NPY_PREVIEW:].tolist()}")
    return "\n".join(lines)


def dump_npy_to_md(results, control_results, output_file: Path = OUTPUT_NPY_FILE):
    """Dump every .npy array in results and control_results completely to Markdown."""
    import numpy as np

    npy_files = [p for p in results if p.suffix.lower() == ".npy"] + \
                [p for p in control_results if p.suffix.lower() == ".npy"]
    if not npy_files:
        print("No .npy files found to dump.")
        return

    print(f"\nDumping {len(npy_files)} .npy files to Markdown: {output_file.name}")
    t0 = time.time()

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Project Compilation — Full NPY Data Dump\n\n")
        out.write(f"- **Generated:** {date.today().isoformat()}\n")
        out.write(f"- **Project root:** `{ROOT}`\n")
        out.write(f"- **Total .npy files:** {len(npy_files)}\n\n")
        out.write(
            "This document contains the complete, unabbreviated array contents of all `.npy` files "
            "from the experiment and validation results.\n\n"
        )

        out.write("## Index\n\n")
        out.write("| # | Relative Path | Shape | Dtype | Elements |\n")
        out.write("|---|---------------|-------|-------|----------|\n")

        file_info = []
        for i, path in enumerate(npy_files, 1):
            rel = path.relative_to(ROOT).as_posix()
            try:
                arr = np.load(path, allow_pickle=True)
                shape = str(arr.shape)
                dtype = str(arr.dtype)
                size = str(arr.size)
                file_info.append((i, path, rel, arr))
            except Exception as exc:
                shape, dtype, size = "ERROR", "ERROR", str(exc)
                file_info.append((i, path, rel, None))
            out.write(f"| {i} | [`{rel}`](#{i}-{rel.replace('/', '-').replace('.', '-')}) | `{shape}` | `{dtype}` | `{size}` |\n")

        out.write("\n---\n\n")

        for i, path, rel, arr in file_info:
            anchor = f"{i}-{rel.replace('/', '-').replace('.', '-')}"
            out.write(f'<a id="{anchor}"></a>\n\n')
            out.write(f"## {i}. `{rel}`\n\n")
            if arr is None:
                out.write("> **Error:** Could not load array from file.\n\n")
                continue

            out.write(f"- **Relative Path:** `{rel}`\n")
            out.write(f"- **Shape:** `{arr.shape}`\n")
            out.write(f"- **Dtype:** `{arr.dtype}`\n")
            out.write(f"- **Elements:** `{arr.size}`\n")
            if arr.size and arr.dtype.kind in "biuf":
                out.write(f"- **Min:** `{arr.min():.6g}` | **Max:** `{arr.max():.6g}` | **Mean:** `{arr.mean():.6g}`\n")
            out.write("\n```json\n")

            if arr.ndim == 2:
                out.write("[\n  ")
                out.write(",\n  ".join(json.dumps(row.tolist()) for row in arr))
                out.write("\n]\n")
            elif arr.ndim <= 1:
                out.write(json.dumps(arr.tolist()))
                out.write("\n")
            else:
                out.write(json.dumps(arr.tolist()))
                out.write("\n")

            out.write("```\n\n---\n\n")

    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"NPY Markdown dump complete ({len(npy_files)} files, {size_mb:.2f} MB in {elapsed:.2f}s).")
    print(f"Saved to:\n  {output_file}\n")


def read_result_text(path: Path):
    text = read_code(path)
    return compact_json(text) if path.suffix.lower() == ".json" else text


def read_code(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def wrap_code(code: str):
    """Hard-wrap long lines (PDF has no horizontal scroll), keeping indent."""
    out = []
    for line in code.expandtabs(4).splitlines():
        if len(line) <= WRAP_WIDTH:
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        out.extend(textwrap.wrap(
            line, WRAP_WIDTH, subsequent_indent=" " * (indent + 4),
            break_long_words=True, break_on_hyphens=False,
        ) or [""])
    return "\n".join(out)


def register_mono_font():
    """Prefer a Unicode monospace font (arrows, Greek letters in comments);
    fall back to built-in Courier, which cannot draw those characters."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                pdfmetrics.registerFont(TTFont("ProjMono", candidate))
                return "ProjMono", True
            except Exception:
                continue
    return "Courier", False


def build_pdf(files, results, control_results):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
    )

    mono_name, unicode_ok = register_mono_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], fontSize=20)
    body = styles["BodyText"]
    mono = ParagraphStyle("Mono", fontName=mono_name, fontSize=FONT_SIZE,
                          leading=FONT_SIZE + 1.6)
    listing = ParagraphStyle("List", fontName=mono_name, fontSize=8,
                             leading=10.5)
    file_header = ParagraphStyle(
        "FH", fontName="Helvetica-Bold", fontSize=10, spaceBefore=4,
        spaceAfter=6, textColor=colors.HexColor("#7a1f1f"))
    result_header = ParagraphStyle(
        "RH", parent=file_header, fontSize=8, spaceBefore=8, spaceAfter=3,
        textColor=colors.HexColor("#1f3a7a"), keepWithNext=1)

    def esc(text):
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not unicode_ok:
            text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    def footer(canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawString(0.6 * inch, 0.4 * inch,
                              f"Project code compilation - {date.today().isoformat()}")
        canvas_obj.drawRightString(A4[0] - 0.6 * inch, 0.4 * inch, f"p.{doc.page}")
        canvas_obj.restoreState()

    # context.md (project memory) goes first, then all code files.
    entries = ([CONTEXT_FILE] if CONTEXT_FILE.is_file() else []) + list(files)
    relative = [f.relative_to(ROOT).as_posix() for f in entries]

    story = [
        Spacer(1, 1.2 * inch),
        Paragraph("Project Code Compilation<br/>Unified Benchmark of Grokking Predictors", title),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Generated: {date.today().isoformat()}", body),
        Paragraph(f"Project root: {esc(str(ROOT))}", body),
        Paragraph(f"Total code files: <b>{len(files)}</b>", body),
        Paragraph(f"Result files (after the code): <b>{len(results)}</b> "
                  "from 08_Experiments/results (checkpoints *.pt not included; "
                  ".npy summarized here, full dump in project_compilation_npy_dump.md)", body),
        Paragraph(f"Controlled-experiment result files (after that): <b>{len(control_results)}</b> "
                  "from controls_results (synthetic-control validation of the predictor criterion)", body),
        Spacer(1, 0.2 * inch),
        Paragraph("Contents", styles["Heading2"]),
        Preformatted(esc("\n".join(f"{i:>3}. {name}" for i, name in enumerate(relative, 1))), listing),
        PageBreak(),
    ]

    for index, (path, rel) in enumerate(zip(entries, relative), 1):
        print(f"Processing: {rel}")
        code = wrap_code(read_code(path)) or "(empty file)"
        story.append(Paragraph(f"{index}. {esc(rel)}", file_header))
        story.append(Preformatted(esc(code), mono))
        story.append(PageBreak())

    max_width = A4[0] - 1.2 * inch
    max_height = A4[1] - 2.0 * inch

    def add_results_section(section_title, files_list, base_dir, description):
        if not files_list:
            return
        story.append(Paragraph(section_title, title))
        story.append(Paragraph(
            f"{len(files_list)} files from {esc(base_dir.relative_to(ROOT).as_posix())}. "
            + description, body))
        story.append(Spacer(1, 0.2 * inch))
        for path in files_list:
            rel = path.relative_to(base_dir).as_posix()
            print(f"Processing result: {rel}")
            story.append(Paragraph(esc(rel), result_header))
            ext = path.suffix.lower()
            if ext == ".png":
                img = Image(str(path))
                scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
                img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
                story.append(img)
                continue
            text = summarize_npy(path) if ext == ".npy" else read_result_text(path)
            story.append(Preformatted(esc(wrap_code(text) or "(empty file)"), mono))

    add_results_section(
        "Results", results, RESULTS_DIR,
        "Text files are shown as they are (JSON number lists put on one line). "
        ".npy arrays are summarised (shape, dtype, min/max/mean, values). "
        "Plots are embedded as images. Model checkpoints (*.pt) are not included.")

    add_results_section(
        "Controlled Experiment Results", control_results, CONTROLS_DIR,
        "Synthetic-control validation of the pre-grok predictor evaluation criterion "
        "(run_control_experiment.py): summary_table.csv, detailed_results.json, "
        "and the sensitivity / null-distribution plots.")

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE), pagesize=A4,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Project Code Compilation",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    files = collect_files()
    results = collect_results(RESULTS_DIR)
    control_results = collect_results(CONTROLS_DIR)
    if not files:
        print("No code files found.")
        return

    # Generate full NPY markdown dump alongside PDF
    dump_npy_to_md(results, control_results)

    try:
        build_pdf(files, results, control_results)
    except ImportError:
        print("reportlab is not installed. Run: pip install reportlab")
        return

    print(f"\nDone! {len(files)} code files, {len(results)} result files, "
          f"and {len(control_results)} controlled-experiment result files compiled.")
    print(f"Outputs saved to:\n  PDF:    {OUTPUT_FILE}\n  NPY MD: {OUTPUT_NPY_FILE}")


if __name__ == "__main__":
    main()
