"""Clarel (1876), Melville's 18,000-line poem, is not on Project Gutenberg but is
transcribed on English Wikisource as Clarel/Part N/Canto M. Fetch every canto
in order, strip the wiki markup, write corpus/melville/clarel.txt.
"""
import json, pathlib, re, time, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "persona-transformers/0.1 (research; contact via github 2016judea)"}
API = "https://en.wikisource.org/w/api.php?"


def api(**params):
    params.update(format="json")
    url = API + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(5 * (attempt + 1))     # Wikisource rate-limits bursts; back off
    raise RuntimeError("rate limited six times: " + url)


def pages():
    out, cont = [], {}
    while True:
        d = api(action="query", list="allpages", apprefix="Clarel/", apnamespace=0, aplimit=500, **cont)
        out += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


def strip(wt):
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)          # templates (header, nop, etc.), one level
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", wt)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"'''?", "", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", "", wt, flags=re.M)
    wt = re.sub(r"^[:;*#]+", "", wt, flags=re.M)
    wt = re.sub(r"[ \t]+$", "", wt, flags=re.M)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def key(title):
    m = re.search(r"Part (\d+)/Canto (\d+)", title)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 99)


def main():
    titles = [t for t in pages() if re.search(r"Part \d+/Canto \d+$", t)]
    titles.sort(key=key)
    print(f"{len(titles)} cantos listed")
    parts = []
    for t in titles:
        d = api(action="parse", page=t, prop="wikitext")
        body = strip(d["parse"]["wikitext"]["*"])
        if len(body) < 200:
            print("  thin:", t, len(body))
        parts.append(body)
        time.sleep(1.5)
    text = "\n\n\n".join(parts) + "\n"
    dest = ROOT / "corpus" / "melville" / "clarel.txt"
    dest.write_text(text)
    print(f"wrote {dest.name}: {len(text)} chars, {text.count(chr(10))} lines")


if __name__ == "__main__":
    main()
