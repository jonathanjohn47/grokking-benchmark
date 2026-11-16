"""compile_python_files.py
=======================

Compiles absolutely everything across the grokking-benchmark project into:
1. `project_compilation.pdf` — Single comprehensive PDF document (excluding only .pdf and .npy).
2. `project_compilation.md`  — Lightweight Markdown twin of the PDF compilation (greatly reduced space).
3. `project_compilation_npy_dump.md` — Complete, unabbreviated contents of every .npy array in results.

Usage:
    python 06_Code/scripts/compile_python_files.py
"""

import json
import re
import textwrap
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "project_compilation.pdf"
OUTPUT_MD_FILE = ROOT / "project_compilation.md"
OUTPUT_NPY_FILE = ROOT / "project_compilation_npy_dump.md"
CONTEXT_FILE = ROOT / "context.md"

SKIP_DIRS = {
    "__pycache__", ".git", ".idea", ".vs", ".vscode", ".claude",
    ".venv", "venv", "env", "build", "dist", "node_modules",
    ".pytest_cache", ".mypy_cache", "graphify-out", ".mplconfig",
}

EXCLUDE_EXTS = {".pdf", ".npy"}
EXCLUDE_NAMES = {".DS_Store", OUTPUT_FILE.name, OUTPUT_MD_FILE.name, OUTPUT_NPY_FILE.name}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
RESULTS_DIR = ROOT / "08_Experiments" / "results"
CONTROLS_DIR = ROOT / "controls_results"

WRAP_WIDTH = 112
FONT_SIZE = 7


def should_skip(path: Path):
    try:
        rel = path.relative_to(ROOT)
        return any(part in SKIP_DIRS for part in rel.parts)
    except Exception:
        return True


def collect_all_files():
    """Collect all files across the project excluding .pdf, .npy, and internal skip dirs."""
    text_files = []
    image_files = []
    pt_files = []

    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or should_skip(p):
            continue
        if p.name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_EXTS:
            continue

        ext = p.suffix.lower()
        if ext in {".pt", ".pth"}:
            pt_files.append(p)
        elif ext in IMAGE_EXTENSIONS:
            image_files.append(p)
        else:
            text_files.append(p)

    return text_files, image_files, pt_files


def collect_npy_files():
    """Collect all .npy files across results and the repository for the markdown dump."""
    return sorted(
        p for p in ROOT.rglob("*.npy")
        if p.is_file() and not should_skip(p)
    )


def compact_json(text: str):
    """Put each innermost list on one line (content unchanged, far fewer pages)."""
    return re.sub(
        r"\[([^\[\]{}]*)\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)).strip() + "]",
        text,
    )


def read_docx_text(path: Path):
    """Extract readable text from a .docx file."""
    try:
        with zipfile.ZipFile(path) as z:
            tree = ET.fromstring(z.read("word/document.xml"))
            paragraphs = []
            for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                p_text = "".join(
                    node.text for node in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                    if node.text
                )
                if p_text.strip():
                    paragraphs.append(p_text)
            return "\n\n".join(paragraphs) if paragraphs else "(empty document)"
    except Exception as e:
        return f"(Could not extract docx text: {e})"


def read_file_content(path: Path):
    """Read file content as text, handling docx and encodings gracefully."""
    ext = path.suffix.lower()
    if ext == ".docx":
        return read_docx_text(path)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as exc:
            return f"(Binary or unreadable file: {exc})"

    text = text.replace("\x00", "")
    if ext == ".json":
        text = compact_json(text)
    return text


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


def dump_npy_to_md(npy_files, output_file: Path = OUTPUT_NPY_FILE):
    """Dump every .npy array in results and across the repo completely to Markdown."""
    import numpy as np

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


def make_anchor(text: str) -> str:
    """Generate a clean markdown anchor."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", text).strip("-")


def get_markdown_lang(ext: str) -> str:
    """Map file extension to markdown syntax highlighter language."""
    mapping = {
        ".py": "python",
        ".json": "json",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".tex": "latex",
        ".csv": "csv",
        ".sh": "bash",
        ".zsh": "bash",
        ".html": "html",
        ".xml": "xml",
        ".toml": "toml",
        ".ini": "ini",
        ".txt": "text",
    }
    return mapping.get(ext, "")


def build_markdown_compilation(text_files, image_files, pt_files, output_file: Path = OUTPUT_MD_FILE):
    """Generate a lightweight, full Markdown compilation of everything in the PDF."""
    print(f"\nGenerating Markdown compilation: {output_file.name}")
    t0 = time.time()

    pt_by_dir = defaultdict(list)
    for p in pt_files:
        pt_by_dir[p.parent].append(p)

    total_pt_mb = sum(p.stat().st_size for p in pt_files) / (1024 * 1024)

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Project Comprehensive Compilation\n")
        out.write("## Unified Benchmark of Grokking Predictors in Neural Networks\n\n")
        out.write(f"- **Generated:** {date.today().isoformat()}\n")
        out.write(f"- **Project Root:** `{ROOT}`\n")
        out.write(f"- **Companion PDF:** [`{OUTPUT_FILE.name}`](./{OUTPUT_FILE.name})\n")
        out.write(f"- **Companion NPY Array Dump:** [`{OUTPUT_NPY_FILE.name}`](./{OUTPUT_NPY_FILE.name})\n\n")

        out.write("### Scope & Inclusions\n\n")
        out.write(
            "This document compiles *absolutely everything* across the repository in lightweight Markdown format, "
            "excluding only standalone `.pdf` files and raw `.npy` arrays (which are dumped in `project_compilation_npy_dump.md`):\n\n"
        )
        out.write(f"- **Part I — Documentation, Code & Structured Files:** {len(text_files)} files.\n")
        out.write(f"- **Part II — Visual Figures & Diagnostic Plots:** {len(image_files)} images (linked relatively with live Markdown image embeds).\n")
        out.write(f"- **Part III — Model Checkpoints Inventory:** {len(pt_files)} PyTorch `.pt` files across {len(pt_by_dir)} suites ({total_pt_mb:.1f} MB total).\n\n")

        out.write("---\n\n")
        out.write("## Table of Contents\n\n")
        out.write("### Part I: Documentation, Source Code & Data Files\n\n")
        for i, p in enumerate(text_files, 1):
            rel = p.relative_to(ROOT).as_posix()
            anchor = f"file-{i}-{make_anchor(rel)}"
            out.write(f"{i}. [`{rel}`](#{anchor})\n")

        out.write("\n### Part II: Visual Figures & Diagnostic Plots\n\n")
        for i, p in enumerate(image_files, 1):
            rel = p.relative_to(ROOT).as_posix()
            anchor = f"fig-{i}-{make_anchor(rel)}"
            out.write(f"{i}. [`{rel}`](#{anchor})\n")

        out.write("\n### Part III: Model Checkpoints Inventory (.pt)\n\n")
        out.write("- [Model Checkpoints Table](#part-iii-model-checkpoints-inventory-pt)\n\n")

        out.write("---\n\n")

        # Part I
        out.write("## Part I: Project Documentation, Source Code & Data Files\n\n")
        for i, p in enumerate(text_files, 1):
            rel = p.relative_to(ROOT).as_posix()
            anchor = f"file-{i}-{make_anchor(rel)}"
            lang = get_markdown_lang(p.suffix.lower())
            content = read_file_content(p)
            out.write(f'<a id="{anchor}"></a>\n\n')
            out.write(f"### {i}. `{rel}`\n\n")
            out.write(f"- **Path:** `{rel}`\n")
            out.write(f"- **Size:** {p.stat().st_size:,} bytes\n\n")
            fence = "````" if "```" in content else "```"
            out.write(f"{fence}{lang}\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write(f"{fence}\n\n---\n\n")

        # Part II
        out.write("## Part II: Visual Figures & Diagnostic Plots\n\n")
        for i, p in enumerate(image_files, 1):
            rel = p.relative_to(ROOT).as_posix()
            anchor = f"fig-{i}-{make_anchor(rel)}"
            out.write(f'<a id="{anchor}"></a>\n\n')
            out.write(f"### Fig {i}. `{rel}`\n\n")
            out.write(f"- **Path:** `{rel}`\n")
            out.write(f"- **Size:** {p.stat().st_size:,} bytes\n\n")
            out.write(f"![Fig {i}: {rel}]({rel})\n\n---\n\n")

        # Part III
        out.write("## Part III: Model Checkpoints Inventory (.pt)\n\n")
        out.write(
            f"Total checkpoint files: **{len(pt_files)}** across {len(pt_by_dir)} suites "
            f"({total_pt_mb:.1f} MB total).\n\n"
        )
        out.write("| Suite / Directory | Files | Total Size | Checkpoint Span |\n")
        out.write("|---|---|---|---|\n")
        for parent_dir, flist in sorted(pt_by_dir.items(), key=lambda item: item[0].as_posix()):
            rel_dir = parent_dir.relative_to(ROOT).as_posix()
            flist_sorted = sorted(flist, key=lambda f: f.name)
            total_sz = sum(f.stat().st_size for f in flist) / (1024 * 1024)
            span = f"`{flist_sorted[0].name}` ... `{flist_sorted[-1].name}`" if len(flist_sorted) > 1 else f"`{flist_sorted[0].name}`"
            out.write(f"| `{rel_dir}` | {len(flist)} | {total_sz:.1f} MB | {span} |\n")
        out.write("\n")

    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Markdown compilation complete ({size_mb:.2f} MB in {elapsed:.2f}s).")
    print(f"Saved to:\n  {output_file}\n")


def register_mono_font():
    """Prefer a Unicode monospace font; fall back to built-in Courier."""
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


def build_pdf(text_files, image_files, pt_files):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    mono_name, unicode_ok = register_mono_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], fontSize=20, leading=24)
    subtitle = ParagraphStyle("ST", parent=styles["Heading2"], fontSize=13, spaceBefore=6, textColor=colors.HexColor("#1f3a7a"))
    body = styles["BodyText"]
    mono = ParagraphStyle("Mono", fontName=mono_name, fontSize=FONT_SIZE, leading=FONT_SIZE + 1.6)
    listing = ParagraphStyle("List", fontName=mono_name, fontSize=7.5, leading=9.5)
    file_header = ParagraphStyle(
        "FH", fontName="Helvetica-Bold", fontSize=9, spaceBefore=6,
        spaceAfter=4, textColor=colors.HexColor("#7a1f1f"), keepWithNext=1,
    )
    section_title_style = ParagraphStyle(
        "SecT", fontName="Helvetica-Bold", fontSize=14, spaceBefore=12,
        spaceAfter=8, textColor=colors.HexColor("#1b7f4e"),
    )

    def esc(text):
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not unicode_ok:
            text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    def footer(canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawString(0.6 * inch, 0.4 * inch,
                              f"Project Comprehensive Compilation - {date.today().isoformat()}")
        canvas_obj.drawRightString(A4[0] - 0.6 * inch, 0.4 * inch, f"p.{doc.page}")
        canvas_obj.restoreState()

    story = [
        Spacer(1, 1.0 * inch),
        Paragraph("Project Comprehensive Compilation<br/>Unified Benchmark of Grokking Predictors", title),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Generated:</b> {date.today().isoformat()}", body),
        Paragraph(f"<b>Project root:</b> {esc(str(ROOT))}", body),
        Spacer(1, 0.1 * inch),
        Paragraph("<b>Scope & Inclusions:</b> This single document compiles <i>absolutely everything</i> across the repository that is not in standalone PDF format or NPY array format:", body),
        Paragraph(f"• <b>Text, Code, Docs & Structured Data:</b> {len(text_files)} files (Context memory, markdown, source code, configs, test suites, progress reports, literature notes, LaTeX chapters, JSON summaries, CSVs, logs).", body),
        Paragraph(f"• <b>Visual Plots & Figures:</b> {len(image_files)} images embedded directly (diagnostic plots, training trajectories, model architectures, paper figures).", body),
        Paragraph(f"• <b>Model Checkpoints:</b> {len(pt_files)} PyTorch `.pt` files cataloged and inventoried with checkpoint epochs and disk sizes.", body),
        Paragraph("• <i>Note:</i> .npy arrays are compiled unabbreviated in <code>project_compilation_npy_dump.md</code>; standalone .pdf files are excluded to prevent recursive duplication.", body),
        Spacer(1, 0.2 * inch),
        Paragraph("Contents / File Index", subtitle),
    ]

    # Index listing of all text and image files
    all_indexed = [(p, p.relative_to(ROOT).as_posix()) for p in text_files + image_files]
    index_lines = [f"{i:>3}. {rel}" for i, (_, rel) in enumerate(all_indexed, 1)]
    story.append(Preformatted(esc("\n".join(index_lines)), listing))
    story.append(PageBreak())

    # Section 1: Text, Code, Markdown & Data Files
    story.append(Paragraph("Part I: Project Documentation, Source Code & Data Files", section_title_style))
    story.append(Paragraph(f"Total files: {len(text_files)}", body))
    story.append(Spacer(1, 0.1 * inch))

    for index, path in enumerate(text_files, 1):
        rel = path.relative_to(ROOT).as_posix()
        print(f"Processing ({index}/{len(text_files)}): {rel}")
        content = read_file_content(path)
        wrapped = wrap_code(content) or "(empty file)"
        story.append(Paragraph(f"{index}. {esc(rel)}", file_header))
        story.append(Preformatted(esc(wrapped), mono))
        story.append(PageBreak())

    # Section 2: Visual Figures & Diagnostic Plots
    max_width = A4[0] - 1.2 * inch
    max_height = A4[1] - 2.0 * inch

    story.append(Paragraph("Part II: Visual Figures & Diagnostic Plots", section_title_style))
    story.append(Paragraph(f"Total visual images: {len(image_files)}", body))
    story.append(Spacer(1, 0.1 * inch))

    for index, path in enumerate(image_files, 1):
        rel = path.relative_to(ROOT).as_posix()
        print(f"Processing image ({index}/{len(image_files)}): {rel}")
        story.append(Paragraph(f"Fig {index}. {esc(rel)}", file_header))
        try:
            img = Image(str(path))
            scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            story.append(img)
        except Exception as exc:
            story.append(Paragraph(f"(Could not render image: {exc})", body))
        story.append(PageBreak())

    # Section 3: Model Checkpoints Inventory
    story.append(Paragraph("Part III: Model Checkpoints Inventory (.pt)", section_title_style))
    pt_by_dir = defaultdict(list)
    for p in pt_files:
        pt_by_dir[p.parent].append(p)

    story.append(Paragraph(
        f"Total checkpoint files: <b>{len(pt_files)}</b> across {len(pt_by_dir)} suites "
        f"({sum(p.stat().st_size for p in pt_files) / (1024 * 1024):.1f} MB total).", body
    ))
    story.append(Spacer(1, 0.1 * inch))

    table_data = [["Suite / Directory", "Files", "Total Size", "Checkpoint Span"]]
    for parent_dir, flist in sorted(pt_by_dir.items(), key=lambda item: item[0].as_posix()):
        rel_dir = parent_dir.relative_to(ROOT).as_posix()
        flist_sorted = sorted(flist, key=lambda f: f.name)
        total_sz = sum(f.stat().st_size for f in flist) / (1024 * 1024)
        span = f"{flist_sorted[0].name} ... {flist_sorted[-1].name}" if len(flist_sorted) > 1 else flist_sorted[0].name
        table_data.append([rel_dir, str(len(flist)), f"{total_sz:.1f} MB", span])

    t = Table(table_data, colWidths=[2.6 * inch, 0.7 * inch, 1.0 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3a7a")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d0d0")),
    ]))
    story.append(t)

    print(f"\nBuilding PDF document: {OUTPUT_FILE} ...")
    doc = SimpleDocTemplate(
        str(OUTPUT_FILE), pagesize=A4,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Project Comprehensive Compilation",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    print(f"Scanning repository at: {ROOT}")
    text_files, image_files, pt_files = collect_all_files()
    npy_files = collect_npy_files()

    print(f"Collected for PDF compilation:")
    print(f"  - Text, docs & code files : {len(text_files)}")
    print(f"  - Visual images           : {len(image_files)}")
    print(f"  - Checkpoints (.pt)       : {len(pt_files)}")
    print(f"  - NPY arrays (for MD dump): {len(npy_files)}")

    # 1. Dump all NPY files to Markdown
    dump_npy_to_md(npy_files)

    # 2. Build Lightweight Markdown Compilation of the Project
    build_markdown_compilation(text_files, image_files, pt_files)

    # 3. Build Comprehensive PDF
    try:
        build_pdf(text_files, image_files, pt_files)
    except ImportError:
        print("reportlab is not installed. Run: pip install reportlab")
        return

    pdf_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    md_size_mb = OUTPUT_MD_FILE.stat().st_size / (1024 * 1024)
    npy_size_mb = OUTPUT_NPY_FILE.stat().st_size / (1024 * 1024)

    print(f"\nDone! Comprehensive outputs successfully generated.")
    print(f"Outputs:")
    print(f"  PDF Compilation : {OUTPUT_FILE} ({pdf_size_mb:.2f} MB)")
    print(f"  MD Compilation  : {OUTPUT_MD_FILE} ({md_size_mb:.2f} MB)")
    print(f"  NPY Dump        : {OUTPUT_NPY_FILE} ({npy_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
