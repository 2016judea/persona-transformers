# Reading notes — what the literature changes about this project

Grounded: 2026-10-02. Each entry: the source, the claim that matters here, and
what it changes. Abstracts and project pages were read directly; where only an
abstract was reachable, that is said.

## A. Training small models on small corpora

**Muennighoff et al. 2023, *Scaling Data-Constrained Language Models*,
arxiv.org/abs/2305.16264.** Repeating data for up to ~4 epochs costs almost
nothing against fresh data; past that, returns fall off fast and extra compute
buys little. Mitigations when data is scarce: more epochs up to that point, not
more parameters.
*Here:* our runs are far past this. 5,000 iterations × 64 × 256 chars = 82M
chars seen. Shakespeare's train split is 4.8M, McCarthy's 5.2M, Melville's 6.9M:
**12–17 epochs**. The train/val gap (1.07 vs 1.33 nats on Shakespeare) is the
memorisation that implies, and memorised passages are a confound for every
"what does the model believe" probe. Experiment: a 4-epoch run per author
(≈1,200 iterations for Shakespeare) and compare val loss; expect close.

**Hoffmann et al. 2022, *Chinchilla*, arxiv.org/abs/2203.15556.** Compute-optimal
training scales tokens and parameters together; Chinchilla (70B params, 1.4T
tokens) sets the ratio near **20 tokens per parameter**.
*Here:* 5M character-tokens is compute-optimal for a ~250K-parameter model. Our
10.66M model is ~40× over-parameterised for its data. That is the data-constrained
regime Muennighoff describes, and it is why a much smaller model is worth
trying, not just cheaper.

**Eldan & Li 2023, *TinyStories*, arxiv.org/abs/2305.07759.** Models under 10M
parameters, even one transformer block deep, write fluent multi-paragraph prose
when the data distribution is narrow. Quality was graded by GPT-4 against a
rubric rather than by loss alone.
*Here:* one author is a narrow distribution. Supports a 1–2 layer model per
author, which is also the regime where circuits are legible (see Elhage). And
licenses an LLM-judge evaluation of samples beyond bits/char.

**Karpathy, nanoGPT README (github.com/karpathy/nanoGPT).** `shakespeare_char`:
6 layers, 6 heads, 384 wide, 256 context, expected val loss 1.4697 on the 1MB
tiny-shakespeare file; `--device=mps` for Apple silicon. The CPU config is
4 layers, 4 heads, 128 wide, dropout 0, val 1.88.
*Here:* our 1.327 on the 5.4MB complete works is consistent. The CPU config is
the natural "small" variant for the sweep above.

**Jordan et al., modded-nanogpt (github.com/KellerJordan/modded-nanogpt).**
The speedrun's wins, in rough order of transferability to our scale: the Muon
optimizer (orthogonalised momentum, ~1.5× sample-efficiency over Adam), rotary
position embeddings, QK-norm, untied embedding and head, zero-initialised
output projections, ReLU², value embeddings.
*Here:* any of these changes the operator shapes we measure, so adopt them for
all authors at once or not at all. Rotary would also remove the learned
position embedding whose Fourier spectrum was the one weight-level author
difference we found. Not adopted yet; noted.

**Karpathy, nanochat README (github.com/karpathy/nanochat).** One knob, depth;
width, heads, learning rate, horizon and weight decay are all derived from it.
*Here:* the sweep below should be written the same way: `--depth`, everything
else derived, so the authors are compared at matched compute.

## B. Reading an author out of a model

**Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*,
transformer-circuits.pub/2021/framework.** The QK circuit (W_Qᵀ W_K) and the
OV circuit (W_O W_V) are the right objects: low-rank products that are what a
head actually computes. One-layer models implement bigram and skip-trigram
statistics; induction heads need two layers.
*Here:* this is exactly what `investigate.py` factors, so the measurement was
aimed correctly. The absence of induction heads in all four 6-layer models (max
score 0.07) is a real finding about character-level training on these
corpora, not a measurement failure, and worth a loss-curve check for the bump
Olsson describes.

**Olsson et al. 2022, *In-context Learning and Induction Heads*,
arxiv.org/abs/2209.11895.** Induction heads form at a phase change visible as a
bump in the loss curve, and the detection test is repeated random tokens.
*Here:* our test is theirs. Check `out/*/log.jsonl` for a bump; none expected.

**Martin & Mahoney 2019–21, *Heavy-Tailed Self-Regularization*,
arxiv.org/abs/1901.08276, and the WeightWatcher README.** Fit a power law to
the tail of the eigenvalue spectrum of WᵀW per layer. Alpha near 2 marks the
best-trained layers; below 2 is over-trained and calls for early stopping;
above ~6 the spectrum is random (untrained); above 8 discard. Smaller alpha
correlates with better generalisation, with no test data needed.
*Here:* **this is the closest real theory to the "transform that moulds the
model" question.** Our earlier "decay exponent" (Zipf slope of singular values)
is not their alpha. `investigate.py` now fits the real one with the `powerlaw`
package. Prediction, given the layout result: alpha will track training length
and corpus size, not author. If it tracks author, that is the first weight-level
author signal and worth a second look.

**Cilibrasi & Vitányi 2005, *Clustering by Compression*,
arxiv.org/abs/cs/0312044.** Normalised Compression Distance,
NCD(x,y) = (C(xy) − min(C(x),C(y))) / max(C(x),C(y)), with bzip2, clustered
five Russian novelists (3–4 texts each) perfectly by author. In English
translation the tree partly regrouped by **translator**.
*Here:* our cross-perplexity matrix is the neural version of NCD, and NCD itself
is the no-network baseline the models have to beat. `scripts/stylometry.py`
computes it book by book across all three corpora. The translator effect is a
warning: Gutenberg's Shakespeare carries an editor's modernised spelling and
stage directions, and the LibGen McCarthy files carry a publisher's
typesetting; part of any "author" signal is edition.

**Delétang et al. 2023, *Language Modeling Is Compression*,
arxiv.org/abs/2309.10668.** A model's log-loss is a compression rate; the
equivalence is exact. Chinchilla compresses images and audio it never trained
on better than PNG and FLAC.
*Here:* licenses reading bits/char as a distance, and the essays probe as
"how well does the McCarthy model compress McCarthy it never saw".

**Huh et al. 2024, *The Platonic Representation Hypothesis*,
arxiv.org/abs/2405.07987.** Representations converge across models, modalities
and training as scale grows.
*Here:* consistent with layouts and spectra being identical across authors
while the conditional distributions differ. Not evidence for it; a frame.

**Belrose et al. 2023, *Tuned Lens*, arxiv.org/abs/2303.08112** (and the
logit lens it fixes). Decode the residual stream after each block through the
final head to see when the model commits to a prediction.
*Here:* the plain logit lens needs no training and is now in
`investigate.py`: bits/char after each block, per author. Hypothesis: verse
with speech headings commits earlier than long subordinate prose.

**Burrows 2002, *Delta*, as taught at programminghistorian.org/en/lessons/
introduction-to-stylometry-with-python.** Take the most frequent words
(mostly function words: unconscious, topic-independent), z-score each word's
frequency across texts, and Delta is the mean absolute z-difference. Beats
chi-squared because no single common word dominates.
*Here:* two uses. Corpus-level Delta is a second classical baseline. More
interesting: **Delta from each model's generated text to each real corpus.** If
samples land within Delta reach of their own author and not the others, the
model has captured the unconscious stratum of style, which is the strongest
cheap test of "speaks like the persona".

**Huang et al. 2024, *Can LLMs identify authorship*, arxiv.org/abs/2403.08213.**
LLMs attribute authorship among 10–20 candidates without fine-tuning, and can
be guided by explicit linguistic features.
*Here:* a blind-judge test. Hand an LLM unlabeled samples from the four models,
ask which of the three authors, record the confusion matrix and the tells it
names. Precedent for LLM grading is TinyStories.

**Cunningham et al. 2023 / Bricken et al. 2023, sparse autoencoders**
(arxiv.org/abs/2309.08600; transformer-circuits.pub/2023/monosemantic-features).
Dictionary features over activations are more interpretable than neurons.
*Here:* the right next instrument once a 1–2 layer model per author exists.
Not started.

## What this queues, in order of cost

1. `scripts/stylometry.py`: NCD dendrogram of every book + Burrows' Delta, no GPU. *(built)*
2. `investigate.py`: Martin–Mahoney alpha per layer, logit lens per block. *(built)*
3. Delta and the LLM blind judge on model samples, after the retrains finish.
4. A 4-epoch run per author to measure what memorisation is costing.
5. A `--depth` sweep (1, 2, 4 layers) per author at matched compute, which is
   where circuits become legible and where an SAE would go.
