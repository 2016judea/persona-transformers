"""Assemble a McCarthy corpus from short public excerpts and his own published
essays, because the novels are in copyright and are not fetched.

Sources, in order of trust:
  1. Aidan's own quotations master (Notion pull at
     ~/Desktop/writing-topology/data/personal_corpus/quotes.json), author McCarthy
  2. Wikiquote, Cormac McCarthy page (quotes by him; "about" sections skipped)
  3. Goodreads author quotes, every page
  4. Nautilus: "The Kekulé Problem" and its sequel, his two published essays

Writes corpus/mccarthy/quotes.txt (one excerpt per paragraph, deterministic
shuffle) and corpus/mccarthy/essays.txt. Prints counts only; excerpt text never
goes to stdout.
"""
import html, json, pathlib, random, re, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "corpus" / "mccarthy"
DEST.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/128 Safari/537.36"}
LOCAL = pathlib.Path.home() / "Desktop/writing-topology/data/personal_corpus/quotes.json"


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            err = e; time.sleep(2 + 2 * i)
    print(f"  fetch failed {url}: {err}"); return ""


def clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = t.replace("—", "--").replace("–", "-").replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t).strip()
    return t


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def local_quotes():
    rows = json.load(open(LOCAL))
    out = [r["quote"] for r in rows if "mccarthy" in (r.get("author") or "").lower()]
    print(f"local master: {len(out)}")
    return out


def wikiquote():
    j = get("https://en.wikiquote.org/w/api.php?action=parse&page=Cormac_McCarthy&prop=wikitext&format=json")
    if not j:
        return []
    wt = json.loads(j)["parse"]["wikitext"]["*"]
    out, section = [], ""
    for line in wt.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line)
        if m:
            section = m.group(2).lower(); continue
        if "about" in section or "external" in section or "misattributed" in section:
            continue
        if line.startswith("* ") and not line.startswith("** "):
            q = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", line[2:])
            q = re.sub(r"'''?", "", q)
            q = re.sub(r"<ref[^>]*>.*?</ref>", "", q)
            q = re.sub(r"\{\{[^}]*\}\}", "", q)
            q = clean(q)
            if len(q) > 40:
                out.append(q)
    print(f"wikiquote: {len(out)}")
    return out


def goodreads():
    out, p = [], 1
    while p <= 120:
        h = get(f"https://www.goodreads.com/author/quotes/4178.Cormac_McCarthy?page={p}")
        blocks = re.findall(r'<div class="quoteText">(.*?)<span class="authorOrTitle">', h, re.S)
        if not blocks:
            break
        for b in blocks:
            b = b.split("&#8213;")[0].split("―")[0]
            q = clean(b).strip('"').strip("“” ")
            if len(q) > 20:
                out.append(q)
        p += 1
        time.sleep(1.0)
    print(f"goodreads: {len(out)} across {p-1} pages")
    return out


def nautilus():
    urls = ["https://nautil.us/the-kekule-problem-236574/",
            "https://nautil.us/cormac-mccarthy-returns-to-the-kekule-problem-237058/"]
    parts = []
    for u in urls:
        h = get(u)
        if not h:
            continue
        body = h
        m = re.search(r"<article.*?</article>", h, re.S)
        if m:
            body = m.group(0)
        paras = re.findall(r"<p[^>]*>(.*?)</p>", body, re.S)
        paras = [clean(x) for x in paras]
        paras = [x for x in paras if len(x) > 60 and "nautilus" not in x.lower()[:40]]
        print(f"nautilus {u.split('/')[-2][:40]}: {len(paras)} paragraphs, {sum(map(len, paras))} chars")
        parts.append("\n\n".join(paras))
    return "\n\n\n".join(parts)


def dedupe(quotes):
    quotes = sorted({clean(q) for q in quotes if q}, key=len, reverse=True)
    kept, keys = [], []
    for q in quotes:
        k = norm(q)
        if len(k) < 20:
            continue
        if any(k in kk for kk in keys):       # contained in a longer kept passage
            continue
        kept.append(q); keys.append(k)
    return kept


def main():
    quotes = local_quotes() + wikiquote() + goodreads()
    kept = dedupe(quotes)
    random.Random(1337).shuffle(kept)
    (DEST / "quotes.txt").write_text("\n\n".join(kept) + "\n")
    essays = nautilus()
    (DEST / "essays.txt").write_text(essays + "\n")
    print(f"unique excerpts: {len(kept)}, {sum(map(len, kept))} chars; essays {len(essays)} chars")


if __name__ == "__main__":
    main()
