#!/usr/bin/env python3
"""
Fetches the full Project Gutenberg "Complete Works of William Shakespeare"
and writes a cleaned text file you can feed to cne.py.

Output: shakespeare_complete.txt (UTF-8)
"""

from urllib.request import urlopen
import re

URL = "https://www.gutenberg.org/files/100/100-0.txt"
OUT = "shakespeare_complete.txt"

def strip_gutenberg_boilerplate(text: str) -> str:
    # Remove Gutenberg header/footer blocks if present
    start_pat = re.compile(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .*?\*\*\*", re.IGNORECASE | re.DOTALL)
    end_pat   = re.compile(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .*?\*\*\*", re.IGNORECASE | re.DOTALL)

    start = start_pat.search(text)
    end   = end_pat.search(text)
    if start and end and start.end() < end.start():
        return text[start.end():end.start()]
    return text  # fallback if markers not found

def normalize_whitespace(text: str) -> str:
    # Collapse excessive blank lines; keep line breaks for token window logic
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    print(f"Downloading {URL} …")
    with urlopen(URL) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    print("Downloaded. Cleaning…")
    cleaned = normalize_whitespace(strip_gutenberg_boilerplate(raw))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # quick stats
    import re
    words = re.findall(r"[A-Za-z’']+", cleaned)
    print(f"Wrote {OUT}")
    print(f"Approx stats: {len(cleaned.splitlines()):,} lines, {len(words):,} tokens, {len(set(w.lower() for w in words)):,} unique")

if __name__ == "__main__":
    main()