"""Blind authorship test on model-generated prose (precedent: TinyStories'
GPT-4 grading; Huang et al. 2024 on LLM authorship attribution).

    python3 scripts/blind_judge.py prepare   # out/judge/items.json + prompt.md, labels withheld
    python3 scripts/blind_judge.py score out/judge/answers.json

prepare: K excerpts of L characters from each model's samples, shuffled under
random ids. The judge sees prompt.md only. The key stays in key.json.
score: answers.json is {"id": "shakespeare|melville|mccarthy", ...}; prints the
confusion matrix and accuracy per model. Also accepts a "tells" field per id,
free text, which is collected into tells.md for reading.
"""
import json, pathlib, random, re, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
S, J = ROOT / "out" / "samples", ROOT / "out" / "judge"
K, L = 8, 700
AUTHOR_OF = lambda name: re.sub(r"_(seed\d+|excerpts|v\d+)$", "", name)


def prepare():
    rng = random.Random(7)
    J.mkdir(exist_ok=True)
    items = []
    for p in sorted(S.glob("*.txt")):
        text = p.read_text()
        for _ in range(K):                                  # random windows; verse has short paragraphs
            start = rng.randint(100, len(text) - L - 1)
            cut = text[start:start + L]
            cut = cut[cut.find(" ") + 1: cut.rfind(" ")]   # whole words at both ends
            items.append({"model": p.stem, "author": AUTHOR_OF(p.stem), "text": cut})
    rng.shuffle(items)
    ids = [f"x{i:03d}" for i in range(len(items))]
    key = {i: {"model": it["model"], "author": it["author"]} for i, it in zip(ids, items)}
    (J / "key.json").write_text(json.dumps(key, indent=1))
    prompt = ["Each passage below was written by a small language model trained on exactly one author: "
              "William Shakespeare, Herman Melville, or Cormac McCarthy. For every id, say which author the "
              "model was trained on, and in one short clause name the tell. Answer as JSON: "
              '{"x000": {"author": "...", "tell": "..."}, ...}. Nothing else.\n']
    for i, it in zip(ids, items):
        prompt.append(f"\n### {i}\n{it['text']}\n")
    (J / "prompt.md").write_text("".join(prompt))
    print(f"{len(items)} items from {len(set(it['model'] for it in items))} models -> {J/'prompt.md'}")


def score(path):
    ans = json.loads(pathlib.Path(path).read_text())
    key = json.loads((J / "key.json").read_text())
    conf, per, tells = Counter(), {}, []
    for i, k in key.items():
        a = ans.get(i, {})
        guess = (a.get("author") if isinstance(a, dict) else a) or "?"
        guess = guess.lower().split()[-1] if guess else "?"
        conf[(k["model"], guess)] += 1
        per.setdefault(k["model"], [0, 0])
        per[k["model"]][1] += 1
        if guess == k["author"]:
            per[k["model"]][0] += 1
        if isinstance(a, dict) and a.get("tell"):
            tells.append(f"- {k['model']} → judged {guess}: {a['tell']}")
    print("| model | correct | n | guessed as |\n|---|---:|---:|---|")
    for m, (c, n) in sorted(per.items()):
        g = Counter({x: v for (mm, x), v in conf.items() if mm == m})
        print(f"| {m} | {c} | {n} | " + ", ".join(f"{x} {v}" for x, v in g.most_common()) + " |")
    (J / "tells.md").write_text("\n".join(tells) + "\n")
    print(f"tells -> {J/'tells.md'}")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        score(sys.argv[2])
