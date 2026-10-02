"""Two classical, model-free views of the same books the GPTs were trained on.

1. Normalised Compression Distance (Cilibrasi & Vitanyi 2005), bzip2, every
   book against every book, hierarchical clustering, author purity.
2. Burrows' Delta (2002): z-scored frequencies of the N most frequent words,
   mean absolute difference. Leave-one-out nearest-author accuracy per book.

    python3 scripts/stylometry.py                 # the corpora
    python3 scripts/stylometry.py --samples out/samples   # + model samples as extra "books"

Writes out/stylometry.json, out/fig_ncd.png, out/fig_delta.png. Prints counts
and distances only; never passage text.
"""
import bz2, json, pathlib, re, sys
from collections import Counter
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from corpus_books import books

OUT = ROOT / "out"
CHUNK = 100_000      # equal-length slices: bzip2's block is 900K and NCD degrades past it
N_WORDS = 150


def ncd_matrix(texts):
    n = len(texts)
    C = [len(bz2.compress(t.encode())) for t in texts]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cxy = len(bz2.compress((texts[i] + texts[j]).encode()))
            D[i, j] = D[j, i] = (cxy - min(C[i], C[j])) / max(C[i], C[j])
    return D


def delta_matrix(texts):
    toks = [re.findall(r"[a-z']+", t.lower()) for t in texts]
    pooled = Counter()
    for tk in toks:
        pooled.update(tk)
    vocab = [w for w, _ in pooled.most_common(N_WORDS)]
    F = np.array([[Counter(tk)[w] / max(1, len(tk)) for w in vocab] for tk in toks])
    Z = (F - F.mean(0)) / (F.std(0) + 1e-12)
    n = len(texts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.abs(Z[i] - Z[j]).mean()
    return D, vocab, Z


def purity(D, authors):
    """Leave-one-out: does each book's nearest neighbour share its author?"""
    hits, per = 0, Counter()
    tot = Counter(authors)
    for i in range(len(authors)):
        d = D[i].copy(); d[i] = np.inf
        j = int(d.argmin())
        if authors[j] == authors[i]:
            hits += 1; per[authors[i]] += 1
    return hits / len(authors), {a: per[a] / tot[a] for a in tot}


def author_means(D, authors):
    A = sorted(set(authors))
    M = {}
    for a in A:
        for b in A:
            ia = [i for i, x in enumerate(authors) if x == a]
            ib = [i for i, x in enumerate(authors) if x == b]
            vals = [D[i, j] for i in ia for j in ib if i != j]
            M[f"{a}->{b}"] = float(np.mean(vals)) if vals else None
    return M


def dendrogram(D, labels, authors, title, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram as dg
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), "average")
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(labels) + 1.5))
    colors = {a: c for a, c in zip(sorted(set(authors)), ["#b5542d", "#2d6fb5", "#3f8f3f", "#7a4f9d"])}
    dg(Z, labels=[f"{a[:4]} · {l[:34]}" for a, l in zip(authors, labels)], orientation="left", ax=ax,
       leaf_font_size=7, link_color_func=lambda k: "#777")
    for lbl in ax.get_ymajorticklabels():
        lbl.set_color(colors[next(a for a in colors if lbl.get_text().startswith(a[:4]))])
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    rows = books()
    if "--samples" in sys.argv:
        sdir = pathlib.Path(sys.argv[sys.argv.index("--samples") + 1])
        for p in sorted(sdir.glob("*.txt")):
            rows.append((f"model:{p.stem}", p.stem, p.read_text()))
    authors = [a for a, _, _ in rows]
    labels = [t for _, t, _ in rows]
    texts = [b[len(b) // 3: len(b) // 3 + CHUNK] if len(b) > CHUNK * 1.5 else b[:CHUNK] for _, _, b in rows]
    print(f"{len(rows)} texts: " + ", ".join(f"{a} {authors.count(a)}" for a in sorted(set(authors))))

    Dn = ncd_matrix(texts)
    pn, pern = purity(Dn, authors)
    print(f"NCD   nearest-neighbour author purity {pn:.3f}  " + " ".join(f"{a}={v:.2f}" for a, v in pern.items()))
    Dd, vocab, Z = delta_matrix(texts)
    pd_, perd = purity(Dd, authors)
    print(f"Delta nearest-neighbour author purity {pd_:.3f}  " + " ".join(f"{a}={v:.2f}" for a, v in perd.items()))

    # the words that separate the authors most, by mean z-score per author
    A = sorted(set(a for a in authors if not a.startswith("model:")))
    tells = {}
    for a in A:
        ia = [i for i, x in enumerate(authors) if x == a]
        mz = Z[ia].mean(0)
        order = np.argsort(-mz)
        tells[a] = {"over": [vocab[k] for k in order[:12]], "under": [vocab[k] for k in order[-8:]]}
    res = {"n": len(rows), "authors": authors, "labels": labels,
           "ncd": {"purity": pn, "per_author": pern, "author_means": author_means(Dn, authors), "matrix": Dn.round(4).tolist()},
           "delta": {"purity": pd_, "per_author": perd, "author_means": author_means(Dd, authors), "matrix": Dd.round(4).tolist(),
                     "vocab": vocab, "tells": tells}}
    (OUT / "stylometry.json").write_text(json.dumps(res, indent=1))
    dendrogram(Dn, labels, authors, "Normalised compression distance (bzip2), average linkage", OUT / "fig_ncd.png")
    dendrogram(Dd, labels, authors, f"Burrows' Delta, {N_WORDS} most frequent words, average linkage", OUT / "fig_delta.png")
    for a in A:
        print(f"  {a:12s} over-uses: {' '.join(tells[a]['over'])}")
    print("wrote out/stylometry.json, fig_ncd.png, fig_delta.png")


if __name__ == "__main__":
    main()
