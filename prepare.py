"""Character-level tokenisation, nanoGPT data/shakespeare_char/prepare.py style,
with ONE vocabulary shared by every author so the three models are comparable
weight-for-weight (same embedding rows mean the same character).

Writes data/<author>/{train,val}.bin and data/meta.pkl.
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
    authors = [p.name for p in sorted(CORPUS.iterdir()) if p.is_dir() and any(p.glob("*.txt"))]
    corpora = {a: load(a) for a in authors}
    # Shared vocabulary: every character seen in any author, but drop characters
    # so rare they are noise (fewer than 20 occurrences across everything).
    from collections import Counter
    counts = Counter()
    for t in corpora.values():
        counts.update(t)
    chars = sorted(c for c, n in counts.items() if n >= 20)
    stoi = {c: i for i, c in enumerate(chars)}
    unk = stoi.get("?", 0)
    DATA.mkdir(exist_ok=True)
    meta = {"vocab_size": len(chars), "itos": {i: c for c, i in stoi.items()}, "stoi": stoi, "authors": {}}
    for a, t in corpora.items():
        ids = np.array([stoi.get(c, unk) for c in t], dtype=np.uint16)
        n = len(ids)
        split = int(n * 0.9)
        (DATA / a).mkdir(exist_ok=True)
        ids[:split].tofile(DATA / a / "train.bin")
        ids[split:].tofile(DATA / a / "val.bin")
        meta["authors"][a] = {"chars": n, "train": split, "val": n - split}
        print(f"{a:12s} {n/1e6:6.2f}M chars  train {split/1e6:.2f}M  val {(n-split)/1e6:.2f}M")
    print(f"vocab {len(chars)} shared")
    (DATA / "meta.pkl").write_bytes(pickle.dumps(meta))


if __name__ == "__main__":
    main()
