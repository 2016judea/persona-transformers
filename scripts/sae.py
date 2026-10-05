"""Sparse autoencoders on the 2-layer persona models (Cunningham et al. 2023;
Bricken et al. 2023). One dictionary per author, same recipe, so the three can
be compared as distributions of feature types.

    python3 scripts/sae.py train shakespeare_d2 [--l1 5e-3] [--steps 3000]
    python3 scripts/sae.py analyze                 # every trained SAE -> out/SAE_REPORT.md

Site: residual stream after block 1 of 2 (d=384). Dictionary 8x = 3072.
Loss = MSE + l1 * |f|_1, decoder columns unit-norm, pre-encoder bias (tied).
Trained online: sample windows of the author's train split, run the frozen
model, fit the SAE on those activations. No activations are stored.

Interpretation needs no corpus text:
  * density      how often each feature fires
  * promotes     decoder column -> ln_f -> lm_head: the characters a feature
                 pushes up when it fires (its output side)
  * fires-on     the distribution of the character AT the firing position and
                 the one before (its input side), from held-out windows
Each feature is then typed by those two sides, and the type mix is the author
fingerprint. Top-activating snippets (30 chars) are written to
out/sae/<model>/contexts.json for reading locally; that file is gitignored.
"""
import json, math, pathlib, pickle, sys, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig

OUT, SAE_DIR = ROOT / "out", ROOT / "out" / "sae"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
meta = pickle.loads((ROOT / "data" / "meta.pkl").read_bytes())
itos = meta["itos"]
EXPANSION = 8
torch.manual_seed(0)


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SITE = arg("--site", 1, int)        # residual after block SITE




class SAE(nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / math.sqrt(d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
        self.normalize()

    @torch.no_grad()
    def normalize(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_dec + self.b_dec, f


def load_model(name):
    ck = torch.load(OUT / name / "ckpt.pt", map_location=DEV)
    m = GPT(GPTConfig(**ck["config"])).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck["author"]


def windows(author, split, n, T):
    d = np.memmap(ROOT / "data" / author / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(d) - T - 1, n)
    return torch.stack([torch.from_numpy(d[i:i + T].astype(np.int64)) for i in ix]).to(DEV)


@torch.no_grad()
def acts(model, x, site=None):
    _, _, extra = model(x, return_resid=True)
    return extra["resid"][SITE if site is None else site]   # (B, T, C)


def train(name):
    l1, steps, batch = arg("--l1", 5e-3), arg("--steps", 3000, int), 64
    model, author = load_model(name)
    T, C = model.config.block_size, model.config.n_embd
    with torch.no_grad():
        scale = acts(model, windows(author, "train", 16, T)).pow(2).mean().sqrt().item()  # RMS per dim -> 1
    sae = SAE(C, EXPANSION * C).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    t0 = time.time()
    for it in range(steps + 1):
        x = acts(model, windows(author, "train", batch, T)).reshape(-1, C) / scale
        xh, f = sae(x)
        mse = (xh - x).pow(2).sum(-1).mean()
        loss = mse + l1 * f.abs().sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        # remove the component of the decoder gradient parallel to each column (keeps unit norm meaningful)
        with torch.no_grad():
            g = sae.W_dec.grad
            g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step(); sae.normalize()
        if it % 500 == 0:
            with torch.no_grad():
                ve = 1 - (xh - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
                l0 = (f > 0).float().sum(-1).mean()
            print(json.dumps({"it": it, "mse": round(mse.item(), 4), "L0": round(l0.item(), 1),
                              "var_explained": round(ve.item(), 3), "s": round(time.time() - t0)}), flush=True)
    d = SAE_DIR / (name if SITE == 1 else f"{name}_s{SITE}"); d.mkdir(parents=True, exist_ok=True)
    torch.save({"sae": sae.state_dict(), "scale": scale, "l1": l1, "steps": steps, "site": SITE,
                "d_in": C, "d_sae": EXPANSION * C, "author": author}, d / "sae.pt")
    print("saved", d / "sae.pt")


@torch.no_grad()
def analyze_one(name):
    ck = torch.load(SAE_DIR / name / "sae.pt", map_location=DEV)
    global SITE
    SITE = ck.get("site", 1)
    model, author = load_model(name.rsplit("_s", 1)[0] if "_s" in name[-4:] else name)
    sae = SAE(ck["d_in"], ck["d_sae"]).to(DEV); sae.load_state_dict(ck["sae"]); sae.eval()
    T, C = model.config.block_size, model.config.n_embd
    V = model.config.vocab_size
    D = ck["d_sae"]
    # output side: what each feature promotes, read through the unembedding
    dirs = sae.W_dec * ck["scale"]                                   # (D, C) in residual units
    logits = model.lm_head(model.transformer.ln_f(dirs))            # (D, V)
    base = model.lm_head(model.transformer.ln_f(sae.b_dec * ck["scale"]))
    promote = logits - base
    top_promote = promote.topk(3, dim=1)
    # input side: density, and what character sits at / before the firing position
    n_win, fires, at_char, before_char = 0, torch.zeros(D, device=DEV), torch.zeros(D, V, device=DEV), torch.zeros(D, V, device=DEV)
    best_val, best_ctx = torch.zeros(D, device=DEV), [None] * D
    x_all = []
    for _ in range(12):
        x = windows(author, "val", 32, T)
        f = sae.encode(acts(model, x).reshape(-1, C) / ck["scale"])     # (B*T, D)
        on = f > 0
        fires += on.float().sum(0); n_win += f.shape[0]
        pos_char = x.reshape(-1)                                       # char at position
        prev = torch.roll(x, 1, dims=1); prev[:, 0] = pos_char.view(32, T)[:, 0]; prev_char = prev.reshape(-1)
        at_char.index_add_(1, pos_char, on.float().T)
        before_char.index_add_(1, prev_char, on.float().T)
        # top context per feature (kept local, gitignored)
        v, idx = f.max(0)
        better = v > best_val
        best_val = torch.where(better, v, best_val)
        for j in torch.nonzero(better).flatten().tolist()[:D]:
            p = int(idx[j]); b, t = divmod(p, T)
            s = max(0, t - 29)
            best_ctx[j] = "".join(itos[int(c)] for c in x[b, s:t + 1].tolist())
    density = (fires / n_win).cpu().numpy()
    dead = float((density == 0).mean())
    # type each live feature
    def ent(p):
        p = p / (p.sum() + 1e-9); p = p[p > 0]; return float(-(p * p.log()).sum())
    types = Counter(); rows = []
    SPACE = {" ", "\n"}; PUNCT = set(".,;:!?'\"-()[]_"); UPPER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for j in np.argsort(-density):
        if density[j] == 0:
            continue
        pc = [itos[int(k)] for k in top_promote.indices[j].tolist()]
        ac = at_char[j]; a_top = itos[int(ac.argmax())]; a_ent = ent(ac)
        p1 = pc[0]
        if a_ent < 0.7:                      # fires on (almost) one specific character
            kind = f"after-char"
        elif p1 in SPACE:
            kind = "predicts word-end"
        elif p1 in PUNCT:
            kind = "predicts punctuation"
        elif p1 in UPPER:
            kind = "predicts capital"
        elif p1 in "aeiou":
            kind = "predicts vowel"
        else:
            kind = "predicts consonant"
        types[kind] += 1
        rows.append({"feature": int(j), "density": float(density[j]), "promotes": pc, "fires_on": a_top,
                     "fires_on_entropy": round(a_ent, 2), "type": kind})
    live = len(rows)
    # variance explained on these windows
    x = acts(model, windows(author, "val", 32, T)).reshape(-1, C) / ck["scale"]
    xh, f = sae(x)
    ve = float(1 - (xh - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum())
    l0 = float((f > 0).float().sum(-1).mean())
    (SAE_DIR / name / "contexts.json").write_text(json.dumps(
        {r["feature"]: {"ctx": best_ctx[r["feature"]], **r} for r in rows[:400]}, indent=1))
    return {"model": name, "site": SITE, "author": author, "l1": ck["l1"], "steps": ck["steps"], "d_sae": D, "dead_frac": dead,
            "live": live, "L0": l0, "var_explained": ve, "types": {k: v / live for k, v in types.items()},
            "density_hist": np.histogram(np.log10(density[density > 0]), bins=np.arange(-6, 0.5, 0.5))[0].tolist(),
            "top_features": rows[:40]}


def analyze():
    names = [p.parent.name for p in sorted(SAE_DIR.glob("*/sae.pt"))]
    res = [analyze_one(n) for n in names]
    (SAE_DIR / "sae_summary.json").write_text(json.dumps(res, indent=1))
    L = ["# Sparse autoencoders on the 2-layer persona models\n",
         f"Site: residual stream after the block in the site column (d=384). Dictionary {EXPANSION}×. One per author, same recipe.\n",
         "| model | site | l1 | var. explained | L0 (features/position) | dead | live |\n|---|---:|---:|---:|---:|---:|---:|"]
    for r in res:
        L.append(f"| {r['model']} | {r['site']} | {r['l1']} | {r['var_explained']:.3f} | {r['L0']:.1f} | {r['dead_frac']*100:.0f}% | {r['live']} |")
    kinds = sorted({k for r in res for k in r["types"]})
    L.append("\n## Feature-type mix (share of live features)\n\nTyped by what a feature fires on (input side) and the character its decoder direction promotes through the unembedding (output side).\n")
    L.append("| type | " + " | ".join(r["model"] for r in res) + " |\n|---|" + "---:|" * len(res))
    for k in kinds:
        L.append(f"| {k} | " + " | ".join(f"{r['types'].get(k, 0)*100:.1f}%" for r in res) + " |")
    L.append("\n## Density histogram (log10 firing rate, bins from 1e-6 to 1)\n\n| model | " + " | ".join(f"1e{b/2-6:.1f}" if False else str(i) for i, b in enumerate(range(0, 12))) + " |\n|---|" + "---:|" * 12)
    for r in res:
        L.append(f"| {r['model']} | " + " | ".join(str(v) for v in r["density_hist"]) + " |")
    L.append("\n## Most frequent live features\n")
    for r in res:
        L.append(f"### {r['model']}\n\n| feature | density | fires on | promotes | type |\n|---:|---:|---|---|---|")
        for t in r["top_features"][:15]:
            L.append(f"| {t['feature']} | {t['density']:.3f} | `{t['fires_on']!r}` (H={t['fires_on_entropy']}) | {' '.join(repr(c) for c in t['promotes'])} | {t['type']} |")
        L.append("")
    (OUT / "SAE_REPORT.md").write_text("\n".join(L))
    print("wrote", OUT / "SAE_REPORT.md")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    else:
        analyze()
