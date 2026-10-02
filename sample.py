"""Talk to a persona model.

    python3 sample.py melville --prompt "Call me " --n 600 --temp 0.8
"""
import argparse, pickle, pathlib
import torch
from model import GPT, GPTConfig

ROOT = pathlib.Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("author")
p.add_argument("--prompt", default="\n")
p.add_argument("--n", type=int, default=500)
p.add_argument("--temp", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--seed", type=int, default=None)
a = p.parse_args()
if a.seed is not None:
    torch.manual_seed(a.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
meta = pickle.loads((ROOT / "data" / "meta.pkl").read_bytes())
ck = torch.load(ROOT / "out" / a.author / "ckpt.pt", map_location=dev)
m = GPT(GPTConfig(**ck["config"])).to(dev); m.load_state_dict(ck["model"]); m.eval()
idx = torch.tensor([[meta["stoi"].get(c, 0) for c in a.prompt]], device=dev)
out = m.generate(idx, a.n, temperature=a.temp, top_k=a.top_k)[0].tolist()
print("".join(meta["itos"][i] for i in out))
