#!/usr/bin/env python3

"""
image_collage.py

Combine all images in a folder into a single collage either
horizontally or vertically.

Requirements:
    pip install pillow

Usage:
    python image_collage.py
"""

from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_images(folder):
    images = [
        f for f in sorted(folder.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images


def horizontal_collage(image_paths, output_path):
    images = [Image.open(p).convert("RGB") for p in image_paths]

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    collage = Image.new("RGB", (total_width, max_height), "white")

    x = 0
    for img in images:
        collage.paste(img, (x, 0))
        x += img.width

    collage.save(output_path)
    print(f"\n✅ Saved: {output_path}")


def vertical_collage(image_paths, output_path):
    images = [Image.open(p).convert("RGB") for p in image_paths]

    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    collage = Image.new("RGB", (max_width, total_height), "white")

    y = 0
    for img in images:
        collage.paste(img, (0, y))
        y += img.height

    collage.save(output_path)
    print(f"\n✅ Saved: {output_path}")


def main():

    print("=" * 60)
    print("Image Collage Generator")
    print("=" * 60)

    while True:
        folder = Path(
            input("\nEnter folder containing images:\n> ").strip().strip('"')
        )

        if folder.exists() and folder.is_dir():
            break

        print("❌ Invalid folder.")

    image_paths = get_images(folder)

    if not image_paths:
        print("❌ No images found.")
        return

    print(f"\nFound {len(image_paths)} images.")

    while True:
        mode = input(
            "\nChoose layout:\n"
            "H = Horizontal\n"
            "V = Vertical\n"
            "> "
        ).strip().lower()

        if mode in ("h", "v"):
            break

    output = input(
        "\nOutput file name (leave blank for collage.png):\n> "
    ).strip()

    if not output:
        output = "collage.png"

    output_path = folder / output

    if mode == "h":
        horizontal_collage(image_paths, output_path)
    else:
        vertical_collage(image_paths, output_path)


if __name__ == "__main__":
    main()