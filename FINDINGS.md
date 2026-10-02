# Findings — what the weights say, and what they don't

Grounded: 2026-10-02, four checkpoints in `out/`, every number from
`out/REPORT.md` / `out/investigation.json` (written by `investigate.py`).

## The models

| model | corpus | train chars | best val (bits/char) | seed |
|---|---|---:|---:|---:|
| shakespeare | Complete works (PG #100) | 4.82M | 1.915 | 1337 |
| melville | 13 novels/collections (PG) | 6.91M | 1.747 | 1337 |
| mccarthy | 1,985 public excerpts, mean 330 chars | 0.60M | 1.837 | 1337 |
| shakespeare_seed7 | same as shakespeare | 4.82M | 1.917 | 7 |

Same architecture throughout: nanoGPT `shakespeare_char`, 6 layers, 6 heads,
384 wide, 256-character context, 10.66M params, one shared 92-character vocab.
The seed-7 model exists to answer one question: how much of a "layout" is the
author, and how much is the dice.

## 1. The layout and the spectra belong to the architecture, not the author

**Head layout.** All four models grow the same template: previous-character
heads in layer 0 (4–6 of them), the long-reach heads in layer 1 (mean attention
distance 44–54 chars), mid-range heads after. Measured as mean |difference| over
sorted per-layer head values:

| metric | shakespeare vs seed7 | shakespeare vs melville | shakespeare vs mccarthy |
|---|---:|---:|---:|
| entropy | 0.063 | 0.069 | 0.086 |
| mean distance (chars) | 5.26 | 2.99 | 5.97 |
| prev-token mass | 0.037 | 0.042 | 0.047 |

Changing the seed moves the layout as much as changing the author. On mean
attention distance it moves it *more*. No induction heads form in any model
(max score 0.07), which is expected at this scale and character level.

**Spectra.** QK effective rank 19.9–21.8 of 64; OV 26.0–27.4 of 64; MLP-in
139–148 of 384; decay exponents identical to two decimals. The singular-value
curves lie on top of each other (`fig_spectra.png`). This is the thing the
question was really asking about, and the answer is: the transform is the
same. A 10M-parameter character model trained on English converges to one
operator shape regardless of whose English it is.

Two weight-level differences do survive the seed control, both small and both
with a mundane reading:

- **Token-embedding effective rank**: McCarthy 18.9 vs 30.8–34.2. His corpus
  is excerpts with fewer character types (no stage directions, fewer capitals,
  1,402 characters mapped to `?`), so the embedding needs fewer directions.
  Corpus artefact, not worldview.
- **Position-embedding low-frequency power**: Melville 29% below 8 cycles per
  window vs Shakespeare 24% and 25% (two seeds agree), McCarthy 19%. Hypothesis:
  Melville's paragraph prose has structure at longer periods than Shakespeare's
  ~40-character verse lines. Testable, untested.

## 2. Where the author actually lives: the conditional distribution

**Cross-perplexity** (bits per character, rows are models, columns held-out text):

| | mccarthy | melville | shakespeare |
|---|---:|---:|---:|
| mccarthy model | **1.84** | 3.22 | 3.80 |
| melville model | 3.05 | **1.80** | 2.30 |
| shakespeare model | 2.73 | 2.27 | **1.87** |
| shakespeare_seed7 | 3.04 | 2.26 | 1.87 |

Melville and Shakespeare are mutually ~0.45 bits more foreign than each is to
itself, and symmetrically so. McCarthy is further from both, and furthest from
Shakespeare (3.80). **Seed caveat:** the two Shakespeare seeds read McCarthy at
2.73 and 3.04, so off-diagonal differences under ~0.3 bits are noise.

**The essays probe** is the cleanest result in the study. McCarthy's two
Nautilus essays were held out of everything. The excerpt-trained McCarthy model
reads them at 2.56 bits/char; Melville's model 3.17, Shakespeare's 3.23 and
3.52. A model that saw only 330-character fragments generalises to his
continuous expository prose and finds it ~0.6–1.0 bits/char more natural than
the full-novel models do. The register transferred.

**The worldview probe** asks each model what follows a shared prompt, 96
samples per prompt, counting the first word. Pooled over eight prompts:

| model | negation first (not/no/nothing/never/none) | "the" first |
|---|---:|---:|
| mccarthy | **169** | 44 |
| melville | 64 | **80** |
| shakespeare | 89 | 38 |
| shakespeare_seed7 | 70 | 53 |

McCarthy's model negates two to three times as often as the others. After
"The sea is" it answered *not* or *no* 48 times in 96 and *nothing* 5 more;
after "There is no" its top word was *god*. Melville's model reaches for the
definite article, the definitional, essayistic move ("Love is the…" 22 of 96,
"Death is the…" 17), and after "There is no" says *doubt*. Shakespeare's two
seeds agree on *more* after "There is no" (13 and 23 of 96), on *not* as the
top word after "God is", and *dead* recurs across prompts in both. Those
seed-stable patterns are the author; the ones that flip between seeds are not.

## 3. What this does and does not license

- The models speak in each register: Shakespeare's emit speech headings and
  stage directions in verse lineation, Melville's run long subordinate clauses
  in paragraphs, McCarthy's produce unpunctuated polysyndeton. Samples are in
  `REPORT.md`.
- The McCarthy corpus is fragments. Its long-range attention statistics are
  not comparable to the two full-text models, and its lower residual norms
  (34.6 vs 52–53 at the last block) are confounded by its shorter run (2,300
  vs 5,000 iterations).
- Reading negation rate as "McCarthy's nihilism" or Melville's definite
  articles as "Melville's essayism" is a hypothesis these numbers make worth
  testing by reading. It is not something a 10M-parameter model has
  established. The thing that is established: the author is not in the shape
  of the operators, which is shared, but in what the operators are pointed at.

## If this goes further

- A BPE or word-level tokeniser (nanoGPT's GPT-2 path) would make token
  geometry meaningful: nearest neighbours of *God*, *sea*, *blood* per model.
- More seeds per author to put error bars on every off-diagonal.
- A McCarthy novel in plain text, from a copy Aidan owns, dropped in
  `corpus/mccarthy/` for a full-text model on equal footing.
- A logit lens through the six blocks to see at which depth each model
  commits to a continuation.
