# Findings — what the weights say, and what they don't

Grounded: 2026-10-02, six checkpoints in `out/`. Every number is from
`out/REPORT.md`, `out/investigation.json` or `out/stylometry.json`, all written
by scripts in this repo. The reading behind the methods is in `READING.md`.

## The models

| model | corpus | train chars | best val (nats) | seed | note |
|---|---|---:|---:|---:|---|
| shakespeare | Complete works, 40 works | 4.82M | 1.327 | 1337 | |
| shakespeare_seed7 | same | 4.82M | 1.329 | 7 | seed control |
| melville | 14 Gutenberg volumes + Clarel | 7.4M | 1.231 | 1337 | |
| melville_v1 | 13 volumes, no Billy Budd, no Clarel | 6.91M | 1.211 | 1337 | corpus control |
| mccarthy | 12 novels | 5.23M | **1.052** | 1337 | |
| mccarthy_excerpts | 1,985 public quotes, mean 330 chars | 0.60M | 1.274 | 1337 | corpus control |

Same architecture throughout: nanoGPT `shakespeare_char`, 6 layers, 6 heads,
384 wide, 256-char context, 10.66M params, one shared 92-character vocabulary.
Two controls answer two questions: how much of a measurement is the dice (seed),
and how much is which books happened to be in the corpus.

McCarthy's prose is the most predictable of the three, character by character,
by a wide margin: 1.05 nats against Melville's 1.23 and Shakespeare's 1.33.

## 1. The operators are the architecture's. Confirmed three ways.

**Head layout.** Every model grows the same template: previous-character heads
in layer 0, long-reach heads in layer 1, mid-range after. Measured as mean
|difference| over sorted per-layer head values, changing the seed moves the
layout as much as changing the author (entropy 0.063 vs 0.069; mean distance
5.3 vs 3.0 chars; prev-token mass 0.037 vs 0.042). Changing Melville's corpus
moved his layer-1 reach from 44.6 to 34.9 chars, more than the gap to anyone.
No induction heads form in any model (max score 0.08); Elhage's framework
says they need two layers to compose and these character models never built
them.

**Singular-value spectra.** QK effective rank 19.9–21.8 of 64, OV 24.8–27.4,
MLP-in 139–152 of 384; decay exponents identical to two decimals across all
six. The curves lie on top of each other.

**Martin–Mahoney alpha**, the real version of the "spectrum that moulds the
model" question (power law on the tail of each layer's eigenvalue spectrum;
near 2 well-trained, under 2 over-trained, over 6 random):

| | Wq | Wk | Wv | Wo | mlp_in | mlp_out |
|---|---:|---:|---:|---:|---:|---:|
| mccarthy | 2.13 | 2.51 | 2.19 | 2.19 | 1.71 | 1.98 |
| melville | 2.08 | 2.32 | 2.20 | 2.41 | 1.71 | 1.99 |
| shakespeare | 2.12 | 2.31 | 1.84 | 2.07 | 1.70 | 2.13 |
| shakespeare_seed7 | 1.91 | 2.18 | 2.19 | 2.01 | 1.71 | 2.17 |

Every layer sits in the well-trained band. The MLP input matrix is 1.70–1.71
for all six models to the second decimal, slightly over the over-training line,
which is the memorisation Muennighoff predicts at 12–17 epochs. Seed moves Wq
by 0.21; author moves it by 0.05. Alpha tracks the training regime, not the
writer.

## 2. Where the author lives

### Cross-perplexity (bits/char; rows models, columns held-out text)

| | mccarthy | melville | shakespeare |
|---|---:|---:|---:|
| mccarthy model | **1.47** | 2.55 | 3.12 |
| melville model | 2.74 | **1.85** | 2.24 |
| shakespeare model | 2.57 | 2.39 | **1.87** |
| shakespeare_seed7 | 2.80 | 2.40 | 1.87 |

Three things in this table:

- **McCarthy and Shakespeare are the far pair** (3.12 and 2.57); Melville sits
  between. The compression tree and Burrows' Delta on the raw books say the
  same (NCD McCarthy↔Shakespeare 0.992 vs ↔Melville 0.978; Delta 1.52 vs 1.16).
- **The distance is asymmetric.** Shakespeare's model reads McCarthy at 2.57;
  McCarthy's reads Shakespeare at 3.12. A model of the richer distribution
  predicts the sparer one better than the reverse. Melville's model reads
  Shakespeare (2.24) better than Shakespeare's reads Melville (2.39) too.
- **Seed noise on the off-diagonal is ~0.25 bits** (2.57 vs 2.80 on McCarthy
  text). Nothing smaller than that is a finding.

### The essays probe

McCarthy's two Nautilus essays were never in any corpus. Bits/char:

| mccarthy (novels) | mccarthy_excerpts | melville | shakespeare | shakespeare_seed7 |
|---:|---:|---:|---:|---:|
| **2.29** | 2.56 | 3.13 | 3.23 | 3.52 |

The novel model is 0.84 bits/char better on his expository prose than any
other author's model, without seeing a sentence of it. The register
transferred from fiction to essay.

### Logit lens: when the text gets decided

Bits/char if the model stopped after block k, and the share of the total
improvement still outstanding after block 4:

| | b1 | b2 | b3 | b4 | b5 | b6 | remaining after b4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 3.88 | 2.98 | 2.47 | 1.96 | 1.62 | 1.51 | **0.19** |
| shakespeare | 3.98 | 3.32 | 2.91 | 2.47 | 2.04 | 1.90 | 0.27 |
| shakespeare_seed7 | 4.02 | 3.33 | 2.90 | 2.46 | 2.07 | 1.90 | 0.26 |
| melville | 4.16 | 3.44 | 3.02 | 2.59 | 2.16 | 1.92 | **0.30** |
| melville_v1 | 4.14 | 3.49 | 3.04 | 2.60 | 2.16 | 1.93 | 0.30 |

This one survives both controls (seed 0.27/0.26, corpus 0.30/0.30). McCarthy's
text is decided earliest in the stack; Melville's needs the last two blocks
most; Shakespeare in between. It is the only computational-structure signal
in this study that is author-driven, and it is a depth-of-processing claim,
not a shape-of-operator claim: the same six blocks, used to different depths.

### The worldview probe, and what the excerpt corpus got wrong

First word after eight shared prompts, 96 samples each. Negation
(not/no/nothing/never/none) and "the", pooled:

| | negation | "the" |
|---|---:|---:|
| mccarthy (novels) | **48** | 43 |
| mccarthy_excerpts | 150 | 58 |
| melville | 65 | 56 |
| melville_v1 | 75 | 84 |
| shakespeare | 79 | 49 |
| shakespeare_seed7 | 56 | 48 |

The earlier result, "McCarthy's model negates two to three times as often as
the others and answers *There is no* with *god*", came from the excerpt
model and does not survive the novels. The novel model negates least of all
six, and answers *There is no* with *sound*, *longer*, *sign*. **A quote corpus
encodes what readers select, not what the writer wrote**: the aphoristic,
nihilistic McCarthy of Goodreads is the readers' McCarthy. The same lesson
reaches Melville: "the" after a prompt moved from 84 to 56 between corpus
versions, so the "definitional Melville" was also not stable. The one
seed-stable pattern is *There is no more* for Shakespeare (13 and 21 of 96),
which the new Melville model has also picked up (8 and 14).

Conclusion for the probe as a method: first-word counts after a prompt are
dominated by which books went in. Keep them as a smoke test, not a finding.

## 3. What the books say with no model at all

Normalised compression distance (bzip2) and Burrows' Delta (150 most frequent
words), over 67 texts: both put every book nearest another by the same author.
Purity 1.00 for all three. So telling the authors apart is not a test a model
can fail. What the trees add is structure inside each author:

- **McCarthy splits into the Tennessee novels and the Southwest novels**: The
  Orchard Keeper, Outer Dark, Child of God and Suttree on one branch; the
  Border Trilogy, No Country, The Road and The Passenger on the other. Blood
  Meridian is his most outlying book (within-author NCD 0.950), Stella Maris
  next.
- **Shakespeare's tree is the genre map**: the Henry VI and Richard histories
  together, Henry IV and V together, the Roman plays together, Lear, Hamlet,
  Macbeth, Cymbeline, Pericles, Winter's Tale and Tempest together, and the
  Sonnets and Lucrece off on their own.
- **Melville's verse is a different writer**: Clarel, Battle-Pieces and John
  Marr form their own branch (Clarel is his most outlying text, 0.971); Pierre
  and The Confidence-Man pair off from the sea narratives.
- Delta's over-used words, which are the unconscious stratum: McCarthy *up,
  back, out, down, said, about*; Melville *of, long, by, from, upon, who,
  little*; Shakespeare *hath, shall, thou, mine, let*. Motion and speech;
  prepositional subordination; second person and modal.

## 4. What this does and does not license

- The models speak in each register (samples in `REPORT.md`). Whether they do
  so in the stylometric sense, landing inside their author's Delta cluster, is
  the next test (`scripts/stylometry.py --samples`), with a blind judge after.
- 12–17 epochs on each corpus means memorisation; the alpha of 1.70 on every
  MLP input says so. A 4-epoch run per author is queued to price it.
- The McCarthy corpus is the only one not from a public-domain edition, and
  Cilibrasi's translator effect warns that part of any author signal is
  edition and typesetting.
- Reading "commits earliest in the stack" as a claim about McCarthy's mind is
  a hypothesis. What is established: his sentences are resolved with less
  processing than Melville's, in a model given identical capacity for both.

## If this goes further

- Delta and blind judge on samples (built, waiting on generation).
- A `--depth` sweep (1, 2, 4 layers) at matched compute, where circuits become
  legible and a sparse autoencoder would go.
- A word-level tokeniser so token geometry means something.
- More seeds per author to put error bars on every off-diagonal.
