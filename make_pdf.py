import os

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def compile_py_to_pdf(output_filename="combined_code.pdf"):
    """
    Find all Python (.py) files in the current project directory,
    excluding virtual environments and cache/build directories,
    and compile their source code into a single PDF.
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

    # Style used for file names
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading2"],
        fontName="Courier-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=15,
        spaceAfter=5,
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
            for line in lines:

                # Escape characters that have special meaning in
                # ReportLab's Paragraph XML/HTML parser.
                escaped_line = (
                    line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace(" ", "&nbsp;")
                    .rstrip("\n")
                )

                # Convert tabs to four spaces
                escaped_line = escaped_line.replace(
                    "\t",
                    "&nbsp;&nbsp;&nbsp;&nbsp;",
                )

                # Preserve blank lines
                if not escaped_line.strip():
                    story.append(Spacer(1, 4))

                else:
                    story.append(
                        Paragraph(
                            escaped_line,
                            code_style,
                        )
                    )

            # Space between Python files
            story.append(Spacer(1, 15))

    # Generate the PDF
    doc.build(story)

    print()
    print(f"🎉 Successfully created: {output_filename}")


if __name__ == "__main__":
    compile_py_to_pdf()
