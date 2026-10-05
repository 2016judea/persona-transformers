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

## 4. Do the models speak like the persona? Two tests, both passed.

**Stylometry on generated prose.** 256K characters sampled from each model
(64 streams, temperature 0.9, top-k 40, seeded from a newline) were added to
the 67 real books as extra texts. Mean Burrows' Delta from each model's prose
to each author's books:

| model samples | to McCarthy | to Melville | to Shakespeare | nearest real book |
|---|---:|---:|---:|---|
| mccarthy | **0.78** | 1.35 | 1.67 | Cities of the Plain |
| mccarthy_excerpts | 1.12 | 1.19 | 1.51 | The Crossing |
| melville | 1.20 | **0.73** | 1.25 | White-Jacket |
| melville_v1 | 1.19 | **0.74** | 1.35 | Omoo |
| shakespeare | 1.60 | 1.37 | **0.73** | The Merchant of Venice |
| shakespeare_seed7 | 1.60 | 1.40 | **0.72** | As You Like It |

Each model's prose sits inside its author's cluster at the same distance
real books sit from each other (within-author Delta on real books: 0.76–0.83).
Delta is built on the most frequent words, the unconscious stratum of style,
so this is the claim "speaks like the persona" in its strongest cheap form.
NCD agrees on every row. The excerpt model is the exception that proves the
corpus matters: it lands nearest McCarthy, but barely (1.12 vs 1.19 to
Melville), because fragments do not teach the connective tissue.

**Blind judge.** 48 random 700-character windows, 8 per model, shuffled under
random ids, handed to a reader that saw nothing else. 48 of 48 attributed to
the right author, every model and both controls included. The tells it named,
as word counts over its 48 one-clause justifications: for Shakespeare, speech
headings, verse lineation, *exeunt*, *thou/hath/doth*; for Melville, Latinate
vocabulary, nautical nouns, quoted dialogue with *said*; for McCarthy,
unquoted dialogue, *aint/dont/goin*, horses, Spanish. Those are the same
strata Delta's over-used words point at (§3), found independently.

## 5. What this does and does not license

- The models pass both persona tests at the level of surface and function-word
  style. Neither test reaches meaning; a judge and Delta would both pass a
  model that produced McCarthy-shaped nonsense, and at 10M parameters much of
  the output is exactly that.
- 12–17 epochs on each corpus means memorisation; the alpha of 1.70 on every
  MLP input says so. A 4-epoch run per author is queued to price it.
- The McCarthy corpus is the only one not from a public-domain edition, and
  Cilibrasi's translator effect warns that part of any author signal is
  edition and typesetting.
- Reading "commits earliest in the stack" as a claim about McCarthy's mind is
  a hypothesis. What is established: his sentences are resolved with less
  processing than Melville's, in a model given identical capacity for both.

## 6. The sweep: what memorisation cost, and where depth stops paying

Twelve more runs (`scripts/sweep.py`, `out/REPORT_sweep.md`): per author, a
6-layer model stopped at 4 epochs (Muennighoff's near-free bound), and 1-, 2-
and 4-layer models at width 384 on the same 4-epoch token budget.

**Validation bits/char:**

| | 1 layer | 2 | 4 | 6 (4 epochs) | 6 (17 epochs) |
|---|---:|---:|---:|---:|---:|
| shakespeare | 2.433 | 2.316 | 2.239 | 2.207 | **1.915** |
| melville | 2.281 | 2.115 | 1.986 | 1.944 | **1.775** |
| mccarthy | 2.065 | 1.893 | 1.809 | 1.785 | **1.518** |

- **Repetition was not waste.** Going from 4 to 17 epochs bought 0.27–0.29
  bits/char for every author, more than the whole gain from 1 to 6 layers.
  Muennighoff's bound is about compute-optimality when fresh data exists;
  with none, repeating still pays. And the gain generalises: the 17-epoch
  McCarthy model reads his unseen essays at 2.29 against 2.55 for the 4-epoch
  one. The memorisation we flagged is the price of that gain, not instead of it.
- **Alpha crossed the line exactly as predicted, and val kept improving.**
  MLP-input alpha went from 1.94–2.17 at 4 epochs to 1.70 at 17 for all three
  authors. Martin–Mahoney's "under 2 = over-trained" threshold fired, and the
  held-out loss fell anyway. Read alpha as a memorisation gauge, not a stop sign.
- **Depth pays less each step.** From 4 to 6 layers is worth 0.02–0.04
  bits/char at this budget. The author ordering (McCarthy easiest, Shakespeare
  hardest, by the same gaps) holds at every depth.
- **No induction heads at any depth.** Max score 0.008 across all 12 runs,
  including the 2-layer models Elhage says can host them. Character-level
  training on these corpora builds previous-character heads (2–3 per model at
  every depth) and never the copy-forward circuit. The in-context mechanism is
  absent from every persona here; the style lives in n-gram-like circuits.
- **The logit-lens ordering is now stable across seed, corpus version, epoch
  count and depth.** Share of the improvement still outstanding at the
  mid-stack: McCarthy 0.145 (6L, 4 epochs) and 0.425 (4L); Shakespeare 0.202
  and 0.478; Melville 0.244 and 0.519. McCarthy is decided earliest, Melville
  latest, in every model built. That is the one author-driven claim about
  computation this study can make with confidence.
- **The far pair and the asymmetry hold at 4 epochs**: McCarthy's model reads
  Shakespeare at 3.17, Shakespeare's reads McCarthy at 2.61.

## 7. Sparse autoencoders: what each 2-layer model spends its features on

One dictionary per author on the 2-layer model's residual stream after block
1 (d=384, 8× expansion = 3,072 features, λ=3, 5,000 steps, same recipe for
all three; `scripts/sae.py`, `out/SAE_REPORT.md`). Variance explained 0.88–0.90,
about 10 features active per position, 1–2% dead. Features are typed with no
corpus text: the decoder direction is read through the unembedding to get the
character it promotes (output side), and the character it fires on is counted
on held-out windows (input side).

| share of live features | shakespeare | melville | mccarthy |
|---|---:|---:|---:|
| predicts a capital letter | **48.2%** | 0.4% | 1.7% |
| predicts punctuation | **12.3%** | 3.1% | 1.3% |
| predicts a vowel | 7.2% | **32.8%** | 13.9% |
| predicts a consonant | 13.2% | 43.3% | **50.5%** |
| predicts a word end (space/newline) | 5.4% | 5.9% | **13.3%** |
| fires on one specific character | 13.7% | 14.4% | 19.2% |

This is the sharpest author fingerprint in the study, and it has to be read
with its caveat in the same breath:

- **Shakespeare's dictionary is mostly the play's apparatus, and two
  controls say exactly how much.** The strongest capital-predicting features
  fire after a sentence end and a line break: they predict the capital that
  opens the next speech heading or verse line. The same 2-layer model and the
  same dictionary were retrained on two stripped texts:

  | share of features | full text | no line numbers | no speech headings or stage directions |
  |---|---:|---:|---:|
  | predicts a capital | 46–48% | 43% | **10%** |
  | predicts punctuation | 12% | 14% | 8% |
  | predicts a vowel | 7% | 5% | **27%** |
  | predicts a consonant | 13% | 18% | **35%** |

  (The analysis draws fresh held-out windows each run, so shares carry about
  ±2 points of noise.) Gutenberg's numbering was worth about 4 points. The
  31,694 speaker labels and 3,528 stage directions were worth about 36. What
  is left, about 10 points, is the verse line itself, each one capitalised.
  And once the apparatus is gone, Shakespeare's allocation looks like
  Melville's (vowels 27% vs 33%, consonants 35% vs 44%): underneath the
  dramatic form, Early Modern verse and nineteenth-century Latinate prose
  make the same demands on a character model. The 2-layer dictionary was
  seeing the play, not the poet.
- **McCarthy's dictionary is consonants and word ends, almost no punctuation.**
  Half the features predict a consonant and 13% predict the end of a word,
  against 5–6% for the others; 1.3% predict punctuation. Short Anglo-Saxon
  words, unpunctuated dialogue, polysyndeton. The model has learned where
  words stop because in his prose that is the hard call; commas never come.
- **Melville's dictionary is vowels.** A third of his features predict a
  vowel, four times Shakespeare's share and twice McCarthy's. Latinate
  polysyllables alternate vowel and consonant deep inside the word, and the
  model needs features to carry that position. His punctuation share (3.1%)
  sits between the two, as his prose does on every other measure.

Density histograms are near-identical across the three (bimodal, a dense
cluster near 10⁻² and a sparse one near 10⁻⁴), so the dictionaries are the
same shape; what differs is what the features are for. Same verdict as §1
from one level further in: identical machinery, allocated to different work.

## If this goes further

- Delta and blind judge on samples: done (§4).
- An SAE at the second site (after block 2), and the same stripping controls on the 6-layer models.
- A word-level tokeniser so token geometry means something.
- More seeds per author to put error bars on every off-diagonal.
