from pathlib import Path

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
}

OUTPUT_FILE = "project_compilation.md"


def should_skip(path: Path):
    return any(part in SKIP_DIRS for part in path.parts)


def main():
    project_path = input("Enter project folder path: ").strip().strip('"')

    root = Path(project_path)

    if not root.exists() or not root.is_dir():
        print("Invalid project directory.")
        return

    py_files = sorted(
        [
            f
            for f in root.rglob("*.py")
            if not should_skip(f.relative_to(root))
        ]
    )

    if not py_files:
        print("No Python files found.")
        return

    output_path = root / OUTPUT_FILE

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# Python Project Compilation\n\n")
        out.write(f"Project: `{root}`\n\n")
        out.write(f"Total Python Files: **{len(py_files)}**\n\n")
        out.write("---\n\n")

        for file in py_files:
            relative = file.relative_to(root)

            print(f"Processing: {relative}")

            try:
                code = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                code = file.read_text(encoding="latin-1")

            out.write(f"## {relative}\n\n")
            out.write("```python\n")
            out.write(code.rstrip())
            out.write("\n```\n\n")
            out.write("---\n\n")

    print("\nDone!")
    print(f"Output saved to:\n{output_path}")


if __name__ == "__main__":
    main()