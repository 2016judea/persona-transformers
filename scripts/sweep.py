"""Run the queued experiments two at a time (the MPS memory limit on 24 GB).

  _4ep : 6 layers, 4 epochs of the author's train split (Muennighoff's
         near-free repetition bound), everything else the shakespeare_char config
  _d1/_d2/_d4 : 1, 2, 4 layers, width 384, 6 heads, same 4-epoch token budget

Tokens per iteration = batch 64 x block 256 = 16,384 chars, so
iters = 4 * train_chars / 16384. Evaluates every 100 iterations and keeps the
best. Logs to out/<name>.log; checkpoints to out/<name>/ckpt.pt.
"""
import pickle, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
meta = pickle.loads((ROOT / "data" / "meta.pkl").read_bytes())
AUTHORS = ["shakespeare", "melville", "mccarthy"]
WORKERS = 2

jobs = []
for a in AUTHORS:
    iters = round(4 * meta["authors"][a]["train"] / 16384)
    jobs.append((f"{a}_4ep", a, ["--max_iters", str(iters), "--eval_interval", "100", "--tag", "_4ep"]))
    for d in (1, 2, 4):
        jobs.append((f"{a}_d{d}", a, ["--max_iters", str(iters), "--eval_interval", "100", "--n_layer", str(d), "--tag", f"_d{d}"]))

running = []
def launch(name, author, args):
    log = open(ROOT / "out" / f"{name}.log", "w")
    p = subprocess.Popen(["caffeinate", "-i", "-s", sys.executable, "train.py", author] + args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"start {name} ({' '.join(args)})", flush=True)
    return (name, p)

while jobs or running:
    while jobs and len(running) < WORKERS:
        running.append(launch(*jobs.pop(0)))
    time.sleep(15)
    for r in running[:]:
        if r[1].poll() is not None:
            print(f"done  {r[0]} exit {r[1].returncode}", flush=True)
            running.remove(r)
print("SWEEP COMPLETE", flush=True)
