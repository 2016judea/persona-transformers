# persona-transformers

Train one small GPT per author, from scratch, on that author alone, the way
Karpathy does it in nanoGPT (`shakespeare_char`) and *Let's build GPT*. Then open
the trained weights and measure what shaped them.

Personas: **Shakespeare**, **Melville**, **McCarthy**.

## Run

```bash
python3 scripts/fetch_corpus.py   # Shakespeare + Melville from Project Gutenberg
python3 prepare.py                # shared 92-char vocab -> data/<author>/{train,val}.bin
python3 train.py shakespeare      # ~35 min on an M5 Pro (MPS); out/shakespeare/ckpt.pt
python3 train.py melville
python3 investigate.py            # out/REPORT.md, out/fig_*.png, out/investigation.json
python3 sample.py melville --prompt "Call me "
```

## McCarthy

Cormac McCarthy is in copyright, so nothing is fetched for him. Put plain-text
files of books you own in `corpus/mccarthy/*.txt`, then rerun `prepare.py`,
`train.py mccarthy` and `investigate.py`. The pipeline is author-agnostic: any
directory under `corpus/` with `.txt` files becomes a persona.

## What "investigate" measures, and why

A transformer does not contain a Laplace transform. What it contains is a stack
of learned linear operators, and the honest analogue of "the transform that
moulds the model" is the **spectrum** of those operators:

- **QK circuit** per head (`W_Qᵀ W_K`): what pairs of positions a head compares.
- **OV circuit** per head (`W_O W_V`): what a head copies once it has attended.
- **MLP in/out**: the per-position nonlinearity.
- **Token and position embeddings.**

For each we compute singular values, effective rank (how many directions the
operator really uses), and the power-law decay exponent. Low effective rank means
the author's text is served by a few sharp features; high rank means the model
needed many.

The **head layout** is measured on held-out text: per head, attention entropy,
mean attention distance, previous-token mass, first-token-sink mass, and an
induction score (the repeat-a-random-sequence test).

The **cross-perplexity matrix** scores each model on each author's held-out text,
in bits per character. It is the closest thing here to a distance between
worldviews: how surprising Shakespeare is to a mind trained only on Melville.

Position embeddings are the one place a frequency transform genuinely applies;
their power spectrum is reported too.

## What this can and cannot say about philosophy

A 10M-parameter character model learns the statistical texture of a prose
style: sentence length, dialogue vs narration, vocabulary repetition, how far
back a clause depends on its antecedent. Those are real and measurable, and they
are the surface through which a worldview reaches a reader. They are not the
worldview. Read the report as "this is how each author's language is built",
and treat any leap from spectrum to metaphysics as a hypothesis to test by
reading, not a finding.
