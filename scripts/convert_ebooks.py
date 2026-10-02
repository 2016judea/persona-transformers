"""Convert ebook files Aidan supplies into plain text for corpus/<author>/.

    python3 scripts/convert_ebooks.py mccarthy ~/Downloads/*.epub ~/Downloads/*.azw3 ...

epub   -> read the OPF spine, strip the XHTML in reading order
mobi/azw3 -> `mobi` (python package) unpacks to epub or html, then as above
pdf    -> pdftotext (poppler); refused if the text layer is thin (scanned)

Prints per-file counts only; book text never goes to stdout. Output files are
named from the ebook's title metadata. corpus/*/*.txt is gitignored: these
files are never committed or redistributed.
"""
import html, pathlib, re, subprocess, sys, tempfile, zipfile
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
BOILERPLATE = re.compile(r"(copyright|all rights reserved|isbn|library of congress|first (vintage|edition)|"
                         r"printed in|www\.|penguin random house|alfred a\. knopf|cover design|also by cormac)",
                         re.I)


def clean_html(s):
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|h\d|li|blockquote|tr)>", "\n\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def epub_to_text(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile").get("full-path")
        opf = ET.fromstring(z.read(opf_path))
        base = pathlib.PurePosixPath(opf_path).parent
        title = (opf.findtext(".//dc:title", namespaces=NS) or pathlib.Path(path).stem).strip()
        items = {i.get("id"): i.get("href") for i in opf.findall(".//opf:manifest/opf:item", NS)}
        order = [items[r.get("idref")] for r in opf.findall(".//opf:spine/opf:itemref", NS) if r.get("idref") in items]
        chunks = []
        for href in order:
            p = str(base / href) if str(base) != "." else href
            p = p.split("#")[0]
            if p not in names:
                continue
            raw = z.read(p).decode("utf-8", "replace")
            body = re.search(r"<body[^>]*>(.*)</body>", raw, re.S)
            t = clean_html(body.group(1) if body else raw)
            if len(t) < 400 and BOILERPLATE.search(t):
                continue  # title/copyright pages
            chunks.append(t)
    return title, "\n\n".join(chunks)


def mobi_to_text(path):
    import mobi
    tmp, out = mobi.extract(str(path))
    out = pathlib.Path(out)
    if out.suffix == ".epub":
        return epub_to_text(out)
    raw = out.read_text("utf-8", "replace")
    body = re.search(r"<body[^>]*>(.*)</body>", raw, re.S)
    t = clean_html(body.group(1) if body else raw)
    m = re.search(r"<title>(.*?)</title>", raw, re.S)
    return (m.group(1).strip() if m else path.stem), t


def pdf_to_text(path):
    t = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    t = re.sub(r"\f", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    words = len(t.split())
    pages = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", pages).group(1)) if "Pages:" in pages else 1
    if words / n < 60:
        raise RuntimeError(f"thin text layer ({words} words over {n} pages): scanned, needs OCR")
    return path.stem, t.strip()


def slug(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]


def main():
    author, files = sys.argv[1], [pathlib.Path(f) for f in sys.argv[2:]]
    dest = ROOT / "corpus" / author
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            if f.suffix.lower() == ".epub":
                title, text = epub_to_text(f)
            elif f.suffix.lower() in (".mobi", ".azw3", ".azw", ".prc"):
                title, text = mobi_to_text(f)
            elif f.suffix.lower() == ".pdf":
                title, text = pdf_to_text(f)
            else:
                print(f"SKIP {f.name}: unknown type"); continue
        except Exception as e:
            print(f"FAIL {f.name}: {e}"); continue
        out = dest / f"{slug(title)}.txt"
        out.write_text(text + "\n")
        print(f"OK   {out.name}: {len(text):,} chars, {len(text.split()):,} words")


if __name__ == "__main__":
    main()
