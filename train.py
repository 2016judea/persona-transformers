"""Train one persona model. nanoGPT train.py, cut to a single device and the
shakespeare_char config (6 layers, 6 heads, 384 wide, 256 context, ~10M params).

    python3 train.py melville            # writes out/melville/ckpt.pt + log.jsonl

Keeps the checkpoint with the best validation loss, so an overfitting run on a
small corpus still leaves the most general model on disk.
"""
import argparse, json, math, pickle, pathlib, time
import numpy as np
import torch
from model import GPT, GPTConfig

ROOT = pathlib.Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("author")
p.add_argument("--max_iters", type=int, default=5000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--min_lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=100)
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--tag", default="", help="suffix for out/<author><tag>, e.g. _seed7 for a control run")
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = p.parse_args()

torch.manual_seed(args.seed)
meta = pickle.loads((ROOT / "data" / "meta.pkl").read_bytes())
data_dir = ROOT / "data" / args.author
train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
out_dir = ROOT / "out" / (args.author + args.tag)
out_dir.mkdir(parents=True, exist_ok=True)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy(d[i:i + args.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
    return x.to(args.device), y.to(args.device)


config = GPTConfig(block_size=args.block_size, vocab_size=meta["vocab_size"], n_layer=args.n_layer,
                   n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
model = GPT(config).to(args.device)
print(f"{args.author}: {model.num_params()/1e6:.2f}M params, device {args.device}")
opt = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.99))


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / (args.warmup + 1)
    if it > args.max_iters:
        return args.min_lr
    r = (it - args.warmup) / (args.max_iters - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


best_val = float("inf")
log = open(out_dir / "log.jsonl", "w")
t0 = time.time()
for it in range(args.max_iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % args.eval_interval == 0:
        losses = estimate_loss()
        rec = {"iter": it, "train": round(losses["train"], 4), "val": round(losses["val"], 4),
               "lr": lr_at(it), "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": config.__dict__, "iter": it,
                        "val": best_val, "author": args.author, "args": vars(args)}, out_dir / "ckpt.pt")
    if it == args.max_iters:
        break
    X, Y = get_batch("train")
    _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"done. best val {best_val:.4f}  ({(time.time()-t0)/60:.1f} min)")
