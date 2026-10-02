"""Every book in every corpus as (author, title, text), so book-level analyses
(NCD, Delta, per-book model scores) share one view of the data.

Melville and McCarthy are one file per book. Shakespeare is one Gutenberg file
with a contents list; it is split on the work titles that list names.
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
SKIP_DIRS = {"probes", "mccarthy_excerpts"}
PG_TITLES = {  # fetch_corpus.py strips the Gutenberg header, so titles live here
    "pg2701": "Moby-Dick", "pg1900": "Typee", "pg4045": "Omoo", "pg13720": "Mardi vol 1",
    "pg13721": "Mardi vol 2", "pg8118": "Redburn", "pg10712": "White-Jacket", "pg34970": "Pierre",
    "pg12384": "Battle-Pieces (verse)", "pg21816": "The Confidence-Man", "pg15859": "The Piazza Tales",
    "pg12841": "John Marr (verse)", "pg15422": "Israel Potter", "pg76513": "Billy Budd + prose pieces",
    "clarel": "Clarel (verse)",
}


def _split_shakespeare(text):
    lines = text.split("\n")
    try:
        c0 = next(i for i, l in enumerate(lines) if l.strip().lower() == "contents")
    except StopIteration:
        return {"complete works": text}
    titles = []
    for l in lines[c0 + 1:c0 + 80]:
        s = l.strip()
        if not s:
            if titles:
                break
            continue
        titles.append(s)
    # each work begins with its title alone on a line, in caps, after the contents
    starts = []
    for i in range(c0 + len(titles) + 1, len(lines)):
        s = lines[i].strip()
        if s in titles and s not in [t for _, t in starts]:
            starts.append((i, s))
    books = {}
    for k, (i, t) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        books[t.title()] = "\n".join(lines[i + 1:j]).strip()
    return books


def books():
    out = []
    for d in sorted(CORPUS.iterdir()):
        if not d.is_dir() or d.name in SKIP_DIRS:
            continue
        for p in sorted(d.glob("*.txt")):
            t = p.read_text(encoding="utf-8", errors="replace")
            if d.name == "shakespeare":
                for title, body in _split_shakespeare(t).items():
                    if len(body) > 20000:
                        out.append((d.name, title, body))
            else:
                title = PG_TITLES.get(p.stem) or re.sub(r"[_\-]+", " ", p.stem).strip()
                out.append((d.name, title[:50], t))
    return out


if __name__ == "__main__":
    for a, t, b in books():
        print(f"{a:12s} {len(b):>9,}  {t}")
