"""Fetch the public-domain corpora from Project Gutenberg and strip the boilerplate.

Shakespeare: the complete works (PG #100).
Melville: every novel/collection PG carries. Candidate ids are verified by
the title in the header, so a wrong id is skipped loudly rather than silently
training on the wrong author.

McCarthy is in copyright and is NOT fetched. Drop text you own into
corpus/mccarthy/*.txt and the rest of the pipeline treats it like the others.
"""
import re, sys, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent / "corpus"
AUTHORS = {
    "shakespeare": {100: "Shakespeare"},
    "melville": {
        2701: "Melville", 1900: "Melville", 4045: "Melville", 13720: "Melville",
        13721: "Melville", 8118: "Melville", 10712: "Melville", 34970: "Melville",
        12384: "Melville", 21816: "Melville", 15859: "Melville",
        12841: "Melville", 15422: "Melville", 76513: "Melville",
    },
}
START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.S)

def fetch(pgid):
    for url in (f"https://www.gutenberg.org/cache/epub/{pgid}/pg{pgid}.txt",
                f"https://www.gutenberg.org/files/{pgid}/{pgid}-0.txt"):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
    raise last

def strip(raw):
    m = START.search(raw)
    body = raw[m.end():] if m else raw
    body = END.split(body)[0]
    return body.strip() + "\n"

for author, ids in AUTHORS.items():
    for pgid, must in ids.items():
        dest = ROOT / author / f"pg{pgid}.txt"
        if dest.exists():
            continue
        try:
            raw = fetch(pgid)
        except Exception as e:
            print(f"SKIP {pgid}: {e}"); continue
        title = re.search(r"Title:\s*(.+)", raw); auth = re.search(r"Author:\s*(.+)", raw)
        title = title.group(1).strip() if title else "?"; auth = auth.group(1).strip() if auth else "?"
        if must.lower() not in auth.lower():
            print(f"SKIP {pgid}: author is '{auth}' not {must} ({title})"); continue
        dest.write_text(strip(raw)); print(f"OK   {pgid}: {title} — {auth} ({dest.stat().st_size//1024} KB)")
