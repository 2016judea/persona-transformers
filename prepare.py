"""Character-level tokenisation, nanoGPT data/shakespeare_char/prepare.py style,
with ONE vocabulary shared by every author so the three models are comparable
weight-for-weight (same embedding rows mean the same character).

Writes data/<author>/{train,val}.bin and data/meta.pkl.

    python3 prepare.py                 # every author; builds the vocab if none exists
    python3 prepare.py mccarthy        # one author, REUSING the frozen vocab in meta.pkl
    python3 prepare.py --rebuild-vocab # rebuild the vocab (invalidates every checkpoint)

The vocab is frozen once written: a checkpoint's embedding rows are indexed by it,
so rebuilding it silently breaks every model trained before. New characters that
the frozen vocab lacks map to '?'.
"""
import pickle, pathlib, sys, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
CORPUS, DATA = ROOT / "corpus", ROOT / "data"


def load(author):
    texts = []
    for p in sorted((CORPUS / author).glob("*.txt")):
        t = p.read_text(encoding="utf-8", errors="replace")
        t = t.replace("\r\n", "\n")
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        texts.append(t)
    return "\n\n".join(texts)


def main():
    rebuild = "--rebuild-vocab" in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    all_authors = [p.name for p in sorted(CORPUS.iterdir()) if p.is_dir() and any(p.glob("*.txt"))]
    authors = wanted or all_authors
    corpora = {a: load(a) for a in authors}
    DATA.mkdir(exist_ok=True)
    meta_path = DATA / "meta.pkl"
    if meta_path.exists() and not rebuild:
        meta = pickle.loads(meta_path.read_bytes())
        stoi = meta["stoi"]
        print(f"reusing frozen vocab ({len(stoi)})")
    else:
        # Shared vocabulary: every character seen in any author, but drop characters
        # so rare they are noise (fewer than 20 occurrences across everything).
        from collections import Counter
        counts = Counter()
        for a in all_authors:
            counts.update(corpora.get(a) or load(a))
        chars = sorted(c for c, n in counts.items() if n >= 20)
        stoi = {c: i for i, c in enumerate(chars)}
        meta = {"vocab_size": len(chars), "itos": {i: c for c, i in stoi.items()}, "stoi": stoi, "authors": {}}
    unk = stoi.get("?", 0)
    for a, t in corpora.items():
        missing = sum(1 for c in t if c not in stoi)
        if missing:
            print(f"{a}: {missing} chars ({missing/len(t)*100:.3f}%) outside the frozen vocab -> '?'")
        ids = np.array([stoi.get(c, unk) for c in t], dtype=np.uint16)
        n = len(ids)
        split = int(n * 0.9)
        (DATA / a).mkdir(exist_ok=True)
        ids[:split].tofile(DATA / a / "train.bin")
        ids[split:].tofile(DATA / a / "val.bin")
        meta["authors"][a] = {"chars": n, "train": split, "val": n - split}
        print(f"{a:12s} {n/1e6:6.2f}M chars  train {split/1e6:.2f}M  val {(n-split)/1e6:.2f}M")
    print(f"vocab {len(stoi)} shared")
    meta_path.write_bytes(pickle.dumps(meta))


if __name__ == "__main__":
    main()
