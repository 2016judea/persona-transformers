"""Generate a large body of prose from each persona model, for the stylometric
tests (Burrows' Delta against the real corpora; a blind judge).

    python3 scripts/generate_samples.py                  # every model in out/*/ckpt.pt
    python3 scripts/generate_samples.py melville         # one

Writes out/samples/<model>.txt (gitignored), 64 streams × N chars each, started
from a newline so nothing is seeded from a real passage. Prints sizes only.
"""
import pathlib, pickle, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
STREAMS, CHARS, TEMP, TOPK = 64, 4000, 0.9, 40
torch.manual_seed(0)

meta = pickle.loads((ROOT / "data" / "meta.pkl").read_bytes())
itos, stoi = meta["itos"], meta["stoi"]
names = sys.argv[1:] or [p.parent.name for p in sorted(OUT.glob("*/ckpt.pt"))]
(OUT / "samples").mkdir(exist_ok=True)

for name in names:
    ck = torch.load(OUT / name / "ckpt.pt", map_location=DEV)
    m = GPT(GPTConfig(**ck["config"])).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    t0 = time.time()
    idx = torch.full((STREAMS, 1), stoi["\n"], dtype=torch.long, device=DEV)
    with torch.no_grad():
        g = m.generate(idx, CHARS, temperature=TEMP, top_k=TOPK)
    text = "\n\n".join("".join(itos[int(i)] for i in row[1:]) for row in g)
    (OUT / "samples" / f"{name}.txt").write_text(text)
    print(f"{name:20s} {len(text):>9,} chars  {len(text.split()):>7,} words  {time.time()-t0:5.0f}s")
