# Persona transformers — what the weights say

Models: mccarthy_4ep, melville_4ep, shakespeare_4ep, shakespeare_noheads_4ep, shakespeare_nonum_4ep. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.

## Training

| model | best val loss (nats/char) | bits/char | at iter | train chars |
|---|---:|---:|---:|---:|
| mccarthy_4ep | 1.237 | 1.785 | 1200 | 5.23M |
| melville_4ep | 1.348 | 1.944 | 1900 | 8.09M |
| shakespeare_4ep | 1.530 | 2.207 | 1100 | 4.82M |
| shakespeare_noheads_4ep | 1.512 | 2.181 | 1000 | 4.47M |
| shakespeare_nonum_4ep | 1.512 | 2.181 | 1100 | 4.82M |

## Cross-perplexity: how each model reads each author

Rows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).

| | mccarthy_4ep text | melville_4ep text | shakespeare_4ep text | shakespeare_noheads_4ep text | shakespeare_nonum_4ep text |
|---|---:|---:|---:|---:|---:|
| **mccarthy_4ep model** | 1.720 | 2.735 | 3.171 | 3.042 | 3.139 |
| **melville_4ep model** | 2.584 | 2.014 | 2.351 | 2.297 | 2.367 |
| **shakespeare_4ep model** | 2.614 | 2.661 | 2.172 | 2.203 | 2.121 |
| **shakespeare_noheads_4ep model** | 2.470 | 2.644 | 2.290 | 2.171 | 2.229 |
| **shakespeare_nonum_4ep model** | 2.574 | 2.649 | 2.173 | 2.205 | 2.109 |

### Probe texts no model trained on

Bits per character on held-out prose (lower = the model finds it more natural).

| model | mccarthy_essays |
|---|---:|
| **mccarthy_4ep model** | 2.548 |
| **melville_4ep model** | 2.921 |
| **shakespeare_4ep model** | 3.141 |
| **shakespeare_noheads_4ep model** | 3.057 |
| **shakespeare_nonum_4ep model** | 3.086 |

## Attention-head layout (mean over held-out windows)

**mccarthy_4ep** — mean normalised entropy 0.574; mean attention distance 21.0 chars (layer means 4.9, 25.1, 11.4, 16.0, 26.3, 42.3); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L0H5 = 0.01; heads with induction > 0.1: 0.

**melville_4ep** — mean normalised entropy 0.532; mean attention distance 19.8 chars (layer means 5.2, 36.1, 9.4, 13.2, 21.3, 33.6); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H5 = 0.01; heads with induction > 0.1: 0.

**shakespeare_4ep** — mean normalised entropy 0.564; mean attention distance 21.1 chars (layer means 5.5, 44.2, 7.9, 11.1, 22.0, 35.6); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H5 = 0.01; heads with induction > 0.1: 0.

**shakespeare_noheads_4ep** — mean normalised entropy 0.576; mean attention distance 22.9 chars (layer means 6.3, 33.9, 9.1, 15.1, 28.3, 44.9); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L5H0 = 0.01; heads with induction > 0.1: 0.

**shakespeare_nonum_4ep** — mean normalised entropy 0.560; mean attention distance 22.1 chars (layer means 5.3, 45.8, 8.7, 10.2, 24.4, 38.3); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H3 = 0.01; heads with induction > 0.1: 0.

## Spectra of the learned operators

Effective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).

| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy_4ep | 16.2 | 21.4 | 181.9 | 28.0 | -6.32 | -6.33 | -0.40 |
| melville_4ep | 16.6 | 22.7 | 156.4 | 30.3 | -6.31 | -6.32 | -0.45 |
| shakespeare_4ep | 17.2 | 23.7 | 184.6 | 34.4 | -6.31 | -6.33 | -0.40 |
| shakespeare_noheads_4ep | 16.4 | 21.6 | 188.7 | 29.5 | -6.31 | -6.33 | -0.39 |
| shakespeare_nonum_4ep | 17.1 | 23.8 | 183.9 | 33.7 | -6.31 | -6.33 | -0.40 |

## Martin–Mahoney alpha (power-law tail of the eigenvalue spectrum of WᵀW)

Near 2 = well-trained layer; below 2 = over-trained; above 6 = random. Mean over layers.

| model | Wq | Wk | Wv | Wo | mlp_in | mlp_out |
|---|---:|---:|---:|---:|---:|---:|
| mccarthy_4ep | 2.26 | 2.42 | 2.98 | 1.77 | 2.17 | 1.84 |
| melville_4ep | 1.97 | 2.07 | 2.62 | 1.95 | 1.94 | 1.78 |
| shakespeare_4ep | 2.31 | 2.45 | 2.90 | 1.75 | 2.16 | 1.80 |
| shakespeare_noheads_4ep | 2.34 | 2.49 | 2.98 | 1.79 | 2.24 | 1.85 |
| shakespeare_nonum_4ep | 2.32 | 2.45 | 2.93 | 1.75 | 2.16 | 1.80 |

## Logit lens: bits/char if the model stopped after block k

| model | embed | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy_4ep | 16.71 | 3.54 | 2.79 | 2.39 | 2.03 | 1.84 | 1.78 |
| melville_4ep | 19.59 | 3.98 | 3.35 | 2.93 | 2.55 | 2.23 | 2.08 |
| shakespeare_4ep | 16.78 | 3.85 | 3.31 | 2.88 | 2.51 | 2.25 | 2.17 |
| shakespeare_noheads_4ep | 16.03 | 3.84 | 3.25 | 2.85 | 2.47 | 2.27 | 2.21 |
| shakespeare_nonum_4ep | 17.23 | 3.95 | 3.44 | 2.95 | 2.54 | 2.21 | 2.12 |

## Residual stream and position

**mccarthy_4ep** — residual norm by block: 1.0, 8.5, 12.0, 14.9, 18.5, 22.6, 26.7; attention:MLP update ratio per block: 0.37, 0.67, 0.76, 0.50, 0.37, 0.37; position-embedding spectral centroid 35.5 cycles/window, 19% of power below 8 cycles.

**melville_4ep** — residual norm by block: 1.1, 10.2, 14.8, 18.8, 22.8, 27.8, 32.5; attention:MLP update ratio per block: 0.32, 0.75, 0.84, 0.65, 0.43, 0.45; position-embedding spectral centroid 34.1 cycles/window, 19% of power below 8 cycles.

**shakespeare_4ep** — residual norm by block: 1.0, 8.1, 10.4, 13.3, 16.7, 20.7, 25.3; attention:MLP update ratio per block: 0.32, 0.64, 0.96, 0.60, 0.42, 0.40; position-embedding spectral centroid 36.4 cycles/window, 17% of power below 8 cycles.

**shakespeare_noheads_4ep** — residual norm by block: 0.9, 8.5, 11.4, 14.3, 18.0, 22.5, 26.9; attention:MLP update ratio per block: 0.30, 0.66, 0.84, 0.49, 0.35, 0.35; position-embedding spectral centroid 37.6 cycles/window, 17% of power below 8 cycles.

**shakespeare_nonum_4ep** — residual norm by block: 1.0, 7.9, 10.2, 12.9, 16.4, 20.5, 25.2; attention:MLP update ratio per block: 0.33, 0.70, 0.87, 0.64, 0.40, 0.37; position-embedding spectral centroid 36.1 cycles/window, 18% of power below 8 cycles.

## Worldview probe: the first word each model puts after a shared prompt

96 sampled continuations per prompt (temperature 0.9, top-k 40); counts of the first word.

## Samples (temperature 0.8, top-k 40)

### mccarthy_4ep

### melville_4ep

### shakespeare_4ep

### shakespeare_noheads_4ep

### shakespeare_nonum_4ep


## Figures

- `fig_cross_bpc.png`
- `fig_cross_bpc_controls.png`
- `fig_cross_bpc_sweep.png`
- `fig_delta.png`
- `fig_eff_rank.png`
- `fig_eff_rank_controls.png`
- `fig_eff_rank_sweep.png`
- `fig_head_layout.png`
- `fig_head_layout_controls.png`
- `fig_head_layout_sweep.png`
- `fig_induction.png`
- `fig_induction_controls.png`
- `fig_induction_sweep.png`
- `fig_ncd.png`
- `fig_resid_positional.png`
- `fig_resid_positional_controls.png`
- `fig_resid_positional_sweep.png`
- `fig_spectra.png`
- `fig_spectra_controls.png`
- `fig_spectra_sweep.png`