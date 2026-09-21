"""
compile_python_files.py
=======================

Compiles EVERY codeable file under the project root into one single PDF
(`project_compilation.pdf`, written in the project root).

The project root is found from this file's location
(06_Code/scripts/ -> two levels up), so the script works on any machine.

Usage:
    python 06_Code/scripts/compile_python_files.py
"""

import textwrap
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "project_compilation.pdf"

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

WRAP_WIDTH = 112
FONT_SIZE = 7


def should_skip(path: Path):
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def is_code_file(path: Path):
    if path == OUTPUT_FILE:
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


def build_pdf(files):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
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

    relative = [f.relative_to(ROOT).as_posix() for f in files]

    story = [
        Spacer(1, 1.2 * inch),
        Paragraph("Project Code Compilation<br/>Unified Benchmark of Grokking Predictors", title),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Generated: {date.today().isoformat()}", body),
        Paragraph(f"Project root: {esc(str(ROOT))}", body),
        Paragraph(f"Total code files: <b>{len(files)}</b>", body),
        Spacer(1, 0.2 * inch),
        Paragraph("Contents", styles["Heading2"]),
        Preformatted(esc("\n".join(f"{i:>3}. {name}" for i, name in enumerate(relative, 1))), listing),
        PageBreak(),
    ]

    for index, (path, rel) in enumerate(zip(files, relative), 1):
        print(f"Processing: {rel}")
        code = wrap_code(read_code(path)) or "(empty file)"
        story.append(Paragraph(f"{index}. {esc(rel)}", file_header))
        story.append(Preformatted(esc(code), mono))
        story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE), pagesize=A4,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Project Code Compilation",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    files = collect_files()
    if not files:
        print("No code files found.")
        return

    try:
        build_pdf(files)
    except ImportError:
        print("reportlab is not installed. Run: pip install reportlab")
        return

    print(f"\nDone! {len(files)} files compiled.")
    print(f"Output saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
