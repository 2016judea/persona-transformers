"""Every book in every corpus as (author, title, text), so book-level analyses
(NCD, Delta, per-book model scores) share one view of the data.

Melville and McCarthy are one file per book. Shakespeare is one Gutenberg file
with a contents list; it is split on the work titles that list names.
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
SKIP_DIRS = {"probes", "mccarthy_excerpts"}


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
                title = re.sub(r"[_\-]+", " ", p.stem).strip()
                m = re.search(r"Title:\s*(.+)", t[:2000])
                out.append((d.name, (m.group(1).strip() if m else title)[:50], t))
    return out


if __name__ == "__main__":
    for a, t, b in books():
        print(f"{a:12s} {len(b):>9,}  {t}")
