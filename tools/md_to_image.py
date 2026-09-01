#!/usr/bin/env python3

"""
markdown_to_images.py

Usage:
    python markdown_to_images.py

The script will ask for:
1. Absolute path to the Markdown file
2. Output directory (optional)
"""

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PAGE_WIDTH = 1928
PAGE_HEIGHT = 1928

MARGIN_X = 40
MARGIN_Y = 40

FONT_SIZE = 20
LINE_SPACING = 4

BACKGROUND = "white"
FOREGROUND = "black"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
]


# ------------------------------------------------------------

def load_font():
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, FONT_SIZE)
        except Exception:
            pass
    return ImageFont.load_default()


font = load_font()


def chars_per_line():
    dummy = Image.new("RGB", (100, 100))
    draw = ImageDraw.Draw(dummy)

    bbox = draw.textbbox((0, 0), "M", font=font)
    char_width = bbox[2] - bbox[0]

    usable = PAGE_WIDTH - (2 * MARGIN_X)

    return max(20, usable // char_width)


MAX_CHARS = chars_per_line()


def wrap_markdown(md):
    wrapped = []

    for line in md.splitlines():

        if line.strip() == "":
            wrapped.append("")
            continue

        if line.startswith("    ") or line.startswith("\t"):
            wrapped.append(line)
            continue

        wrapped.extend(
            textwrap.wrap(
                line,
                width=MAX_CHARS,
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
        )

    return wrapped


def render_page(lines, page_number, outdir):
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1] + LINE_SPACING

    y = MARGIN_Y

    for line in lines:
        draw.text((MARGIN_X, y), line, fill=FOREGROUND, font=font)
        y += line_height

    image.save(outdir / f"page_{page_number:03}.png")


def main():

    print("=" * 60)
    print(" Markdown → PNG Converter")
    print("=" * 60)

    while True:
        md_path = input("\nEnter absolute path of the Markdown file:\n> ").strip().strip('"')

        md_file = Path(md_path)

        if md_file.exists() and md_file.is_file():
            break

        print("\n❌ File not found. Please try again.")

    output = input(
        "\nEnter output folder (leave blank for automatic):\n> "
    ).strip().strip('"')

    if output:
        outdir = Path(output)
    else:
        outdir = md_file.parent / f"{md_file.stem}_images"

    outdir.mkdir(parents=True, exist_ok=True)

    text = md_file.read_text(encoding="utf-8")

    wrapped = wrap_markdown(text)

    dummy = Image.new("RGB", (100, 100))
    draw = ImageDraw.Draw(dummy)

    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1] + LINE_SPACING

    lines_per_page = (PAGE_HEIGHT - (2 * MARGIN_Y)) // line_height

    page = 1

    for i in range(0, len(wrapped), lines_per_page):
        render_page(
            wrapped[i:i + lines_per_page],
            page,
            outdir,
        )
        page += 1

    print(f"\n✅ Done!")
    print(f"Generated {page - 1} image(s).")
    print(f"Output folder:\n{outdir.resolve()}")


if __name__ == "__main__":
    main()