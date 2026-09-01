import argparse
import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
except ImportError:
    sys.exit("reportlab is required. Install it with: pip install reportlab")


def gather_python_files(src_root):
    root = Path(src_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {root}")
    return sorted(root.rglob("*.py"))


def wrap_text(text, canvas_obj, font_name, font_size, max_width):
    if canvas_obj.stringWidth(text, font_name, font_size) <= max_width:
        return [text]

    words = text.split(" ")
    lines = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if canvas_obj.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
        if canvas_obj.stringWidth(word, font_name, font_size) <= max_width:
            current = word
            continue

        # If a single word is longer than the available width, split it.
        chunk = ""
        for char in word:
            if canvas_obj.stringWidth(chunk + char, font_name, font_size) <= max_width:
                chunk += char
            else:
                if chunk:
                    lines.append(chunk)
                chunk = char
        current = chunk

    if current:
        lines.append(current)
    return lines


def compile_python_files_to_pdf(src_root, output_file):
    files = gather_python_files(src_root)
    if not files:
        raise FileNotFoundError(f"No Python files found in: {src_root}")

    output_path = Path(output_file)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    line_height = 12
    font_name = "Courier"
    font_size = 10
    usable_width = width - margin * 2
    y_position = height - margin

    c.setFont(font_name, font_size)

    for file_path in files:
        # Resolve both paths to absolute before computing relative path
        try:
            rel_path = file_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel_path = file_path

        header = f"File: {rel_path}"

        if y_position - line_height * 2 < margin:
            c.showPage()
            c.setFont(font_name, font_size)
            y_position = height - margin

        c.drawString(margin, y_position, header)
        y_position -= line_height * 1.5

        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n").replace("\t", "    ")
                wrapped_lines = wrap_text(line, c, font_name, font_size, usable_width)
                for wrapped_line in wrapped_lines:
                    if y_position - line_height < margin:
                        c.showPage()
                        c.setFont(font_name, font_size)
                        y_position = height - margin
                    c.drawString(margin, y_position, wrapped_line)
                    y_position -= line_height

        y_position -= line_height

    c.save()
    print(f"PDF created: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compile all Python files in the src folder and subfolders into one PDF."
    )
    parser.add_argument(
        "--src",
        default="src",
        help="Source folder containing Python files (default: src)",
    )
    parser.add_argument(
        "--output",
        default="python_files_compiled.pdf",
        help="Output PDF file name (default: python_files_compiled.pdf)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    compile_python_files_to_pdf(args.src, args.output)