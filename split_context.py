import pathlib
p = pathlib.Path("/Users/jonathanjohn/Documents/grokking-benchmark/context.md")
text = p.read_text(encoding="utf-8", errors="ignore")
# Split into 30k char chunks
chunk_size = 30000
for i in range(0, len(text), chunk_size):
    chunk = text[i:i+chunk_size]
    out = pathlib.Path(f"/Users/jonathanjohn/Documents/grokking-benchmark/context_part_{i//chunk_size:03d}.md")
    out.write_text(chunk, encoding="utf-8")
print(f"Split into {(len(text)+chunk_size-1)//chunk_size} parts, total chars {len(text)}")
# Also extract relevant sections
import re
pattern = r"(Switch from 24 log-spaced|Shift from 100 to 50|save-every-100|GCD justification|transition width|EVAL_EVERY|SAVE_EVERY|100.*epoch|50.*epoch)"
matches = []
for m in re.finditer(pattern, text, re.IGNORECASE):
    start = max(0, m.start()-500)
    end = min(len(text), m.end()+1000)
    matches.append(text[start:end])
Pathlib = pathlib.Path("/Users/jonathanjohn/Documents/grokking-benchmark/context_relevant.txt")
Pathlib.write_text("\n\n---MATCH---\n\n".join(matches[-100:]), encoding="utf-8")
print(f"Extracted {len(matches)} relevant snippets into context_relevant.txt")
