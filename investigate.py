"""Open the trained persona models and measure what shapes them.

There is no Laplace transform inside a transformer. What a trained GPT has is a
stack of learned linear operators (the QK and OV circuits of each head, the two
MLP matrices, the token and position embeddings), and the nearest honest analogue
of "the transform that moulds the model" is the SPECTRUM of those operators: their
singular values, how fast they decay, how many directions each one actually uses.
That, plus the LAYOUT of the attention heads (what each head attends to), plus how
each model scores every author's held-out text, is what this script measures.

    python3 investigate.py              # all authors with a checkpoint
    -> out/investigation.json, out/fig_*.png, out/REPORT.md

Every number in the report is written by this script from the checkpoints.
"""
import json, math, pickle, pathlib, sys
import numpy as np
import torch
from model import GPT, GPTConfig

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

meta = pickle.loads((ROOT / "data" / "meta.pkl").read_bytes())
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi.get(c, stoi.get("?", 0)) for c in s]
decode = lambda l: "".join(itos[int(i)] for i in l)

AUTHORS = [p.parent.name for p in sorted(OUT.glob("*/ckpt.pt"))]
if "--models" in sys.argv:                       # comma-separated subset
    want = sys.argv[sys.argv.index("--models") + 1].split(",")
    AUTHORS = [a for a in AUTHORS if a in want]
SKIP_SAMPLES = "--skip-samples" in sys.argv      # sampling and the worldview probe are the slow part
TAG = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else ""   # suffix for REPORT/json/figs
if not AUTHORS:
    sys.exit("no checkpoints in out/*/ckpt.pt")
DATA_OF = {a: torch.load(OUT / a / "ckpt.pt", map_location="cpu")["author"] for a in AUTHORS}

PROMPTS = ["God ", "The sea ", "Death ", "And he said, ", "The world is ", "I "]
N_WINDOWS = 48  # held-out windows per author for attention statistics


def load(author):
    ck = torch.load(OUT / author / "ckpt.pt", map_location=DEVICE)
    m = GPT(GPTConfig(**ck["config"])).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def val_windows(author, n, T):
    author = DATA_OF.get(author, author)   # a control run like shakespeare_seed7 reads shakespeare's data
    d = np.memmap(ROOT / "data" / author / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(d) - T - 1, n)
    x = torch.stack([torch.from_numpy(d[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def spectrum(W):
    """Singular values and the summaries used everywhere below."""
    s = torch.linalg.svdvals(W.detach().float().cpu()).numpy()
    p = s ** 2 / (s ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))      # exp(entropy) of the spectrum
    pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())              # participation ratio
    k = max(4, int(len(s) * 0.5))
    i = np.arange(1, k + 1)
    slope = float(np.polyfit(np.log(i), np.log(s[:k] + 1e-12), 1)[0])  # power-law decay exponent
    return {"sv": s.tolist(), "top": float(s[0]), "eff_rank": eff_rank, "participation_ratio": pr,
            "decay_exponent": slope, "frac_top8": float((s[:8] ** 2).sum() / (s ** 2).sum())}


def alpha_of(W):
    """Martin & Mahoney: power-law exponent of the tail of the eigenvalue
    spectrum of W^T W. Near 2 = well-trained; <2 over-trained; >6 random."""
    import powerlaw, logging, warnings
    logging.getLogger("powerlaw").setLevel(logging.ERROR)
    W = W.detach().float().cpu()
    ev = (torch.linalg.svdvals(W) ** 2).numpy()
    ev = ev[ev > 1e-10]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = powerlaw.Fit(ev, verbose=False)
    return float(f.alpha), float(f.xmin), int((ev >= f.xmin).sum())


def alphas(model):
    C = model.config.n_embd
    rows = []
    for li, blk in enumerate(model.transformer.h):
        W = blk.attn.c_attn.weight
        rows.append({"layer": li,
                     "Wq": alpha_of(W[:C])[0], "Wk": alpha_of(W[C:2 * C])[0], "Wv": alpha_of(W[2 * C:])[0],
                     "Wo": alpha_of(blk.attn.c_proj.weight)[0],
                     "mlp_in": alpha_of(blk.mlp.c_fc.weight)[0], "mlp_out": alpha_of(blk.mlp.c_proj.weight)[0]})
    return rows


@torch.no_grad()
def logit_lens(model, x, y):
    """Decode the residual stream after each block through ln_f + lm_head:
    bits/char the model would score if it stopped there. Where the curve drops
    is where the author's text gets decided."""
    _, _, extra = model(x, return_resid=True)
    out = []
    for r in extra["resid"]:
        logits = model.lm_head(model.transformer.ln_f(r))
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        out.append(float(loss) / math.log(2))
    return out


@torch.no_grad()
def head_layout(model, x):
    """Per head: entropy, mean attention distance, previous-token mass, self mass,
    first-token sink mass. Computed on held-out windows of the author's own text."""
    _, _, extra = model(x, return_attn=True)
    B, T = x.shape
    pos = torch.arange(T, device=DEVICE)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()   # query - key
    layers = []
    for att in extra["attn"]:                                   # (B, H, T, T)
        a = att[:, :, 8:, :]                                      # skip the first 8 queries (too few keys)
        ent = -(a * (a + 1e-12).log()).sum(-1)                   # (B,H,T')
        maxent = torch.log(torch.arange(9, T + 1, device=DEVICE).float())
        ent_n = (ent / maxent).mean((0, 2))
        md = (a * dist[8:, :]).sum(-1).mean((0, 2))
        prev = torch.stack([att[:, :, t, t - 1] for t in range(8, T)], -1).mean((0, 2))
        self_ = torch.stack([att[:, :, t, t] for t in range(8, T)], -1).mean((0, 2))
        sink = att[:, :, 8:, 0].mean((0, 2))
        layers.append({"entropy": ent_n.tolist(), "mean_distance": md.tolist(),
                       "prev_token": prev.tolist(), "self": self_.tolist(), "sink": sink.tolist()})
    return layers


@torch.no_grad()
def induction_scores(model, n=16, L=100):
    """Repeat a random sequence twice; an induction head puts mass, at position
    t in the second copy, on the token just after the earlier occurrence (t-L+1)."""
    V = model.config.vocab_size
    seq = torch.randint(0, V, (n, L), device=DEVICE)
    x = torch.cat([seq, seq], 1)
    _, _, extra = model(x, return_attn=True)
    out = []
    for att in extra["attn"]:
        sc = torch.stack([att[:, :, t, t - L + 1] for t in range(L, 2 * L)], -1).mean((0, 2))
        out.append(sc.tolist())
    return out


@torch.no_grad()
def residual_profile(model, x):
    _, _, extra = model(x, return_resid=True)
    norms = [float(r.norm(dim=-1).mean()) for r in extra["resid"]]
    # contribution of attention vs mlp per block, as norm of the update
    contrib = []
    h = extra["resid"][0]
    for blk in model.transformer.h:
        a, _ = blk.attn(blk.ln_1(h))
        h2 = h + a
        m = blk.mlp(blk.ln_2(h2))
        contrib.append({"attn": float(a.norm(dim=-1).mean()), "mlp": float(m.norm(dim=-1).mean())})
        h = h2 + m
    return norms, contrib


def positional_spectrum(model):
    """The one place a frequency transform genuinely applies: learned position
    embeddings. Mean power spectrum over the 256 positions, averaged across dims."""
    W = model.transformer.wpe.weight.detach().float().cpu().numpy()   # (T, C)
    W = W - W.mean(0, keepdims=True)
    P = np.abs(np.fft.rfft(W, axis=0)) ** 2
    P = P.mean(1)
    P = P / P.sum()
    freqs = np.arange(len(P))
    centroid = float((freqs * P).sum())
    return {"power": P.tolist(), "centroid_cycles_per_window": centroid,
            "frac_low_freq_lt8": float(P[:8].sum())}


@torch.no_grad()
def bits_per_char(model, x, y):
    _, loss = model(x, y)
    return float(loss) / math.log(2)


@torch.no_grad()
def probe_bpc(model, text, T):
    """Bits per character over a held-out text no model trained on, in
    non-overlapping windows of the context length."""
    ids = torch.tensor(encode(text), dtype=torch.long)
    n = (len(ids) - 1) // T
    if n == 0:
        return float("nan")
    x = ids[: n * T].view(n, T).to(DEVICE)
    y = ids[1: n * T + 1].view(n, T).to(DEVICE)
    tot = 0.0
    for i in range(0, n, 16):
        _, loss = model(x[i:i + 16], y[i:i + 16])
        tot += float(loss) * min(16, n - i)
    return tot / n / math.log(2)


WORLD_PROMPTS = ["God is ", "Death is ", "The world is ", "A man is ", "Love is ", "The sea is ", "There is no ", "Man is "]


@torch.no_grad()
def worldview(model, n=96, n_chars=48, temperature=0.9):
    """The model as a conditional distribution: after each shared prompt, sample
    many continuations in one batch and count the first word. This is the most
    direct worldview measurement here, because it asks the model what it
    believes follows 'God is', not how its weights are shaped."""
    import re
    from collections import Counter
    out = {}
    for pr in WORLD_PROMPTS:
        idx = torch.tensor([encode(pr)] * n, device=DEVICE)
        g = model.generate(idx, n_chars, temperature=temperature, top_k=40)
        conts = [decode(r[len(pr):].tolist()) for r in g]
        first = Counter()
        for c in conts:
            m = re.match(r"\s*([A-Za-z']+)", c)
            if m:
                first[m.group(1).lower()] += 1
        out[pr] = {"top_first_words": first.most_common(8),
                   "examples": [c.split("\n")[0][:60] for c in conts[:5]]}
    return out


@torch.no_grad()
def samples(model, temperature=0.8, n_chars=400):
    out = {}
    for p in PROMPTS:
        idx = torch.tensor([encode(p)], device=DEVICE)
        g = model.generate(idx, n_chars, temperature=temperature, top_k=40)
        out[p] = decode(g[0].tolist())
    idx = torch.tensor([encode("\n")], device=DEVICE)
    out["<free>"] = decode(model.generate(idx, 800, temperature=temperature, top_k=40)[0].tolist())
    return out


def main():
    models = {a: load(a) for a in AUTHORS}
    T = next(iter(models.values()))[0].config.block_size
    windows = {a: val_windows(a, N_WINDOWS, T) for a in AUTHORS}
    result = {"authors": AUTHORS, "device": DEVICE, "models": {}, "cross_bpc": {}, "probe_bpc": {}}
    probes = {p.stem: p.read_text() for p in sorted((ROOT / "corpus" / "probes").glob("*.txt")) if p.stat().st_size > 1000}

    for a, (m, ck) in models.items():
        print(f"== {a}: iter {ck['iter']} val {ck['val']:.4f}")
        H, C = m.config.n_head, m.config.n_embd
        hs = C // H
        spectra = {"wte": spectrum(m.transformer.wte.weight), "wpe": spectrum(m.transformer.wpe.weight),
                   "layers": []}
        for blk in m.transformer.h:
            W = blk.attn.c_attn.weight.detach()            # (3C, C)
            Wq, Wk, Wv = W[:C], W[C:2 * C], W[2 * C:]
            Wo = blk.attn.c_proj.weight.detach()           # (C, C)
            heads = []
            for h in range(H):
                sl = slice(h * hs, (h + 1) * hs)
                qk = Wq[sl].T @ Wk[sl]                      # (C, C) rank <= hs
                ov = Wo[:, sl] @ Wv[sl]                     # (C, C) rank <= hs
                heads.append({"QK": spectrum(qk), "OV": spectrum(ov)})
            spectra["layers"].append({"heads": heads,
                                      "mlp_in": spectrum(blk.mlp.c_fc.weight),
                                      "mlp_out": spectrum(blk.mlp.c_proj.weight)})
        x, y = windows[a]
        norms, contrib = residual_profile(m, x[:16])
        result["models"][a] = {
            "iter": ck["iter"], "best_val_loss": ck["val"], "params": m.num_params(),
            "train_chars": meta["authors"][DATA_OF[a]]["train"],
            "data": DATA_OF[a], "seed": ck["args"].get("seed"),
            "head_layout": head_layout(m, x),
            "induction": induction_scores(m),
            "spectra": spectra,
            "resid_norms": norms, "block_contrib": contrib,
            "alphas": alphas(m),
            "logit_lens_bpc": logit_lens(m, x[:16], y[:16]),
            "positional": positional_spectrum(m),
            "samples": {} if SKIP_SAMPLES else samples(m),
            "worldview": {} if SKIP_SAMPLES else worldview(m),
        }
        result["cross_bpc"][a] = {b: bits_per_char(m, *windows[b]) for b in AUTHORS}
        result["probe_bpc"][a] = {k: probe_bpc(m, t, T) for k, t in probes.items()}
        print("   bpc on", {b: round(v, 3) for b, v in result["cross_bpc"][a].items()},
              "probes", {k: round(v, 3) for k, v in result["probe_bpc"][a].items()})

    (OUT / f"investigation{TAG}.json").write_text(json.dumps(result, indent=1))
    figures(result)
    report(result)


def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = R["authors"]
    n = len(A)
    # 1. head layouts
    metrics = ["entropy", "mean_distance", "prev_token", "sink"]
    fig, axes = plt.subplots(len(metrics), n, figsize=(3.2 * n, 2.6 * len(metrics)), squeeze=False)
    for j, a in enumerate(A):
        L = R["models"][a]["head_layout"]
        for i, mname in enumerate(metrics):
            M = np.array([l[mname] for l in L])
            ax = axes[i, j]
            vmax = {"entropy": 1, "mean_distance": None, "prev_token": 1, "sink": 1}[mname]
            im = ax.imshow(M, cmap="magma", aspect="auto", vmin=0, vmax=vmax)
            ax.set_title(f"{a} — {mname}", fontsize=9)
            ax.set_xlabel("head"); ax.set_ylabel("layer")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / f"fig_head_layout{TAG}.png", dpi=130); plt.close(fig)
    # 2. induction
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), squeeze=False)
    for j, a in enumerate(A):
        M = np.array(R["models"][a]["induction"])
        im = axes[0, j].imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=max(0.2, M.max()))
        axes[0, j].set_title(f"{a} — induction score"); axes[0, j].set_xlabel("head"); axes[0, j].set_ylabel("layer")
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / f"fig_induction{TAG}.png", dpi=130); plt.close(fig)
    # 3. spectra: QK, OV, MLP per layer
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    for a in A:
        S = R["models"][a]["spectra"]
        qk = np.mean([[h["QK"]["sv"] for h in l["heads"]] for l in S["layers"]], axis=(0, 1))
        ov = np.mean([[h["OV"]["sv"] for h in l["heads"]] for l in S["layers"]], axis=(0, 1))
        ml = np.mean([l["mlp_in"]["sv"] for l in S["layers"]], axis=0)
        for ax, s, t in zip(axes, (qk, ov, ml), ("QK circuit (mean over heads)", "OV circuit (mean over heads)", "MLP in (mean over layers)")):
            s = np.array(s); s = s[s > 1e-6]
            ax.loglog(np.arange(1, len(s) + 1), s / s[0], label=a); ax.set_title(t); ax.set_xlabel("index"); ax.set_ylabel("σ / σ₁")
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f"fig_spectra{TAG}.png", dpi=130); plt.close(fig)
    # 4. effective rank per layer
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for a in A:
        S = R["models"][a]["spectra"]["layers"]
        axes[0].plot([np.mean([h["QK"]["eff_rank"] for h in l["heads"]]) for l in S], marker="o", label=a)
        axes[1].plot([np.mean([h["OV"]["eff_rank"] for h in l["heads"]]) for l in S], marker="o", label=a)
        axes[2].plot([l["mlp_in"]["eff_rank"] for l in S], marker="o", label=a)
    for ax, t in zip(axes, ("QK effective rank", "OV effective rank", "MLP-in effective rank")):
        ax.set_title(t); ax.set_xlabel("layer"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f"fig_eff_rank{TAG}.png", dpi=130); plt.close(fig)
    # 5. residual norms + positional spectrum
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for a in A:
        axes[0].plot(R["models"][a]["resid_norms"], marker="o", label=a)
        P = np.array(R["models"][a]["positional"]["power"])
        axes[1].semilogy(P[:64], label=a)
    axes[0].set_title("residual stream norm after each block"); axes[0].set_xlabel("block"); axes[0].legend()
    axes[1].set_title("position-embedding power spectrum (cycles / 256 chars)"); axes[1].set_xlabel("frequency"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / f"fig_resid_positional{TAG}.png", dpi=130); plt.close(fig)
    # 6. cross bpc
    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.4 * n + 1))
    M = np.array([[R["cross_bpc"][a][b] for b in A] for a in A])
    im = ax.imshow(M, cmap="cividis_r")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{b} text" for b in A]); ax.set_yticks(range(n)); ax.set_yticklabels([f"{a} model" for a in A])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i, j] > M.mean() else "k")
    ax.set_title("bits per character on held-out text"); plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / f"fig_cross_bpc{TAG}.png", dpi=130); plt.close(fig)


def report(R):
    A = R["authors"]
    L = ["# Persona transformers — what the weights say\n",
         f"Models: {', '.join(A)}. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT "
         "shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.\n"]
    L.append("## Training\n\n| model | best val loss (nats/char) | bits/char | at iter | train chars |\n|---|---:|---:|---:|---:|")
    for a in A:
        m = R["models"][a]
        L.append(f"| {a} | {m['best_val_loss']:.3f} | {m['best_val_loss']/math.log(2):.3f} | {m['iter']} | {m['train_chars']/1e6:.2f}M |")
    L.append("\n## Cross-perplexity: how each model reads each author\n\nRows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).\n")
    L.append("| | " + " | ".join(f"{b} text" for b in A) + " |\n|---|" + "---:|" * len(A))
    for a in A:
        L.append(f"| **{a} model** | " + " | ".join(f"{R['cross_bpc'][a][b]:.3f}" for b in A) + " |")
    probes = sorted({k for a in A for k in R["probe_bpc"][a]})
    if probes:
        L.append("\n### Probe texts no model trained on\n\nBits per character on held-out prose (lower = the model finds it more natural).\n")
        L.append("| model | " + " | ".join(probes) + " |\n|---|" + "---:|" * len(probes))
        for a in A:
            L.append(f"| **{a} model** | " + " | ".join(f"{R['probe_bpc'][a][k]:.3f}" for k in probes) + " |")
    L.append("\n## Attention-head layout (mean over held-out windows)\n")
    for a in A:
        m = R["models"][a]
        HL, IND = m["head_layout"], np.array(m["induction"])
        ent = np.array([l["entropy"] for l in HL]); dist = np.array([l["mean_distance"] for l in HL])
        prev = np.array([l["prev_token"] for l in HL]); sink = np.array([l["sink"] for l in HL])
        li, hi = np.unravel_index(IND.argmax(), IND.shape)
        L.append(f"**{a}** — mean normalised entropy {ent.mean():.3f}; mean attention distance {dist.mean():.1f} chars "
                 f"(layer means {', '.join(f'{d:.1f}' for d in dist.mean(1))}); "
                 f"{int((prev > 0.5).sum())} previous-token heads (>0.5 mass); {int((sink > 0.5).sum())} first-token-sink heads; "
                 f"strongest induction head L{li}H{hi} = {IND.max():.2f}; heads with induction > 0.1: {int((IND > 0.1).sum())}.\n")
    L.append("## Spectra of the learned operators\n\nEffective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).\n")
    L.append("| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in A:
        S = R["models"][a]["spectra"]
        qk = np.mean([[h["QK"]["eff_rank"] for h in l["heads"]] for l in S["layers"]])
        ov = np.mean([[h["OV"]["eff_rank"] for h in l["heads"]] for l in S["layers"]])
        ml = np.mean([l["mlp_in"]["eff_rank"] for l in S["layers"]])
        qkd = np.mean([[h["QK"]["decay_exponent"] for h in l["heads"]] for l in S["layers"]])
        ovd = np.mean([[h["OV"]["decay_exponent"] for h in l["heads"]] for l in S["layers"]])
        mld = np.mean([l["mlp_in"]["decay_exponent"] for l in S["layers"]])
        L.append(f"| {a} | {qk:.1f} | {ov:.1f} | {ml:.1f} | {S['wte']['eff_rank']:.1f} | {qkd:.2f} | {ovd:.2f} | {mld:.2f} |")
    L.append("\n## Martin–Mahoney alpha (power-law tail of the eigenvalue spectrum of WᵀW)\n\nNear 2 = well-trained layer; below 2 = over-trained; above 6 = random. Mean over layers.\n")
    L.append("| model | Wq | Wk | Wv | Wo | mlp_in | mlp_out |\n|---|---:|---:|---:|---:|---:|---:|")
    for a in A:
        al = R["models"][a]["alphas"]
        L.append(f"| {a} | " + " | ".join(f"{np.mean([r[k] for r in al]):.2f}" for k in ("Wq", "Wk", "Wv", "Wo", "mlp_in", "mlp_out")) + " |")
    nb = max(len(R["models"][a]["logit_lens_bpc"]) for a in A)
    L.append("\n## Logit lens: bits/char if the model stopped after block k\n\n| model | embed | " + " | ".join(f"b{k}" for k in range(1, nb)) + " |\n|---|" + "---:|" * nb)
    for a in A:
        L.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in R["models"][a]["logit_lens_bpc"]) + " |")
    L.append("\n## Residual stream and position\n")
    for a in A:
        m = R["models"][a]
        c = m["block_contrib"]
        ratios = ", ".join("%.2f" % (b["attn"] / b["mlp"]) for b in c)
        norms = ", ".join("%.1f" % v for v in m["resid_norms"])
        L.append(f"**{a}** — residual norm by block: {norms}; "
                 f"attention:MLP update ratio per block: {ratios}; "
                 f"position-embedding spectral centroid {m['positional']['centroid_cycles_per_window']:.1f} cycles/window, "
                 f"{m['positional']['frac_low_freq_lt8']*100:.0f}% of power below 8 cycles.\n")
    L.append("## Worldview probe: the first word each model puts after a shared prompt\n\n96 sampled continuations per prompt (temperature 0.9, top-k 40); counts of the first word.\n")
    for pr in (WORLD_PROMPTS if not SKIP_SAMPLES else []):
        L.append(f"**`{pr.strip()}`**\n")
        for a in A:
            w = R["models"][a]["worldview"][pr]
            L.append(f"- {a}: " + ", ".join(f"{k} ({v})" for k, v in w["top_first_words"]))
        L.append("")
    L.append("## Samples (temperature 0.8, top-k 40)\n")
    for a in A:
        L.append(f"### {a}\n")
        for p, s in R["models"][a]["samples"].items():
            L.append(f"**prompt `{p!r}`**\n\n```\n{s.strip()}\n```\n")
    L.append("\n## Figures\n\n" + "\n".join(f"- `{f.name}`" for f in sorted(OUT.glob("fig_*.png"))))
    (OUT / f"REPORT{TAG}.md").write_text("\n".join(L))
    print("wrote", OUT / f"REPORT{TAG}.md")


if __name__ == "__main__":
    main()
