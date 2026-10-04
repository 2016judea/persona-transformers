# Persona transformers — what the weights say

Models: mccarthy, mccarthy_4ep, mccarthy_d1, mccarthy_d2, mccarthy_d4, melville, melville_4ep, melville_d1, melville_d2, melville_d4, shakespeare, shakespeare_4ep, shakespeare_d1, shakespeare_d2, shakespeare_d4. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.

## Training

| model | best val loss (nats/char) | bits/char | at iter | train chars |
|---|---:|---:|---:|---:|
| mccarthy | 1.052 | 1.518 | 4750 | 5.23M |
| mccarthy_4ep | 1.237 | 1.785 | 1200 | 5.23M |
| mccarthy_d1 | 1.431 | 2.065 | 1200 | 5.23M |
| mccarthy_d2 | 1.312 | 1.893 | 1200 | 5.23M |
| mccarthy_d4 | 1.254 | 1.809 | 1200 | 5.23M |
| melville | 1.230 | 1.775 | 5000 | 8.09M |
| melville_4ep | 1.348 | 1.944 | 1900 | 8.09M |
| melville_d1 | 1.581 | 2.281 | 1900 | 8.09M |
| melville_d2 | 1.466 | 2.115 | 1900 | 8.09M |
| melville_d4 | 1.376 | 1.986 | 1900 | 8.09M |
| shakespeare | 1.327 | 1.915 | 4750 | 4.82M |
| shakespeare_4ep | 1.530 | 2.207 | 1100 | 4.82M |
| shakespeare_d1 | 1.687 | 2.433 | 1100 | 4.82M |
| shakespeare_d2 | 1.605 | 2.316 | 1100 | 4.82M |
| shakespeare_d4 | 1.552 | 2.239 | 1100 | 4.82M |

## Cross-perplexity: how each model reads each author

Rows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).

| | mccarthy text | mccarthy_4ep text | mccarthy_d1 text | mccarthy_d2 text | mccarthy_d4 text | melville text | melville_4ep text | melville_d1 text | melville_d2 text | melville_d4 text | shakespeare text | shakespeare_4ep text | shakespeare_d1 text | shakespeare_d2 text | shakespeare_d4 text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **mccarthy model** | 1.470 | 1.470 | 1.470 | 1.470 | 1.470 | 2.553 | 2.553 | 2.553 | 2.553 | 2.553 | 3.117 | 3.117 | 3.117 | 3.117 | 3.117 |
| **mccarthy_4ep model** | 1.720 | 1.720 | 1.720 | 1.720 | 1.720 | 2.735 | 2.735 | 2.735 | 2.735 | 2.735 | 3.171 | 3.171 | 3.171 | 3.171 | 3.171 |
| **mccarthy_d1 model** | 1.992 | 1.992 | 1.992 | 1.992 | 1.992 | 2.975 | 2.975 | 2.975 | 2.975 | 2.975 | 3.406 | 3.406 | 3.406 | 3.406 | 3.406 |
| **mccarthy_d2 model** | 1.827 | 1.827 | 1.827 | 1.827 | 1.827 | 2.836 | 2.836 | 2.836 | 2.836 | 2.836 | 3.278 | 3.278 | 3.278 | 3.278 | 3.278 |
| **mccarthy_d4 model** | 1.742 | 1.742 | 1.742 | 1.742 | 1.742 | 2.743 | 2.743 | 2.743 | 2.743 | 2.743 | 3.213 | 3.213 | 3.213 | 3.213 | 3.213 |
| **melville model** | 2.735 | 2.735 | 2.735 | 2.735 | 2.735 | 1.851 | 1.851 | 1.851 | 1.851 | 1.851 | 2.239 | 2.239 | 2.239 | 2.239 | 2.239 |
| **melville_4ep model** | 2.584 | 2.584 | 2.584 | 2.584 | 2.584 | 2.014 | 2.014 | 2.014 | 2.014 | 2.014 | 2.351 | 2.351 | 2.351 | 2.351 | 2.351 |
| **melville_d1 model** | 2.449 | 2.449 | 2.449 | 2.449 | 2.449 | 2.357 | 2.357 | 2.357 | 2.357 | 2.357 | 2.646 | 2.646 | 2.646 | 2.646 | 2.646 |
| **melville_d2 model** | 2.503 | 2.503 | 2.503 | 2.503 | 2.503 | 2.197 | 2.197 | 2.197 | 2.197 | 2.197 | 2.499 | 2.499 | 2.499 | 2.499 | 2.499 |
| **melville_d4 model** | 2.478 | 2.478 | 2.478 | 2.478 | 2.478 | 2.072 | 2.072 | 2.072 | 2.072 | 2.072 | 2.399 | 2.399 | 2.399 | 2.399 | 2.399 |
| **shakespeare model** | 2.568 | 2.568 | 2.568 | 2.568 | 2.568 | 2.394 | 2.394 | 2.394 | 2.394 | 2.394 | 1.868 | 1.868 | 1.868 | 1.868 | 1.868 |
| **shakespeare_4ep model** | 2.614 | 2.614 | 2.614 | 2.614 | 2.614 | 2.661 | 2.661 | 2.661 | 2.661 | 2.661 | 2.172 | 2.172 | 2.172 | 2.172 | 2.172 |
| **shakespeare_d1 model** | 2.663 | 2.663 | 2.663 | 2.663 | 2.663 | 2.931 | 2.931 | 2.931 | 2.931 | 2.931 | 2.406 | 2.406 | 2.406 | 2.406 | 2.406 |
| **shakespeare_d2 model** | 2.629 | 2.629 | 2.629 | 2.629 | 2.629 | 2.766 | 2.766 | 2.766 | 2.766 | 2.766 | 2.272 | 2.272 | 2.272 | 2.272 | 2.272 |
| **shakespeare_d4 model** | 2.605 | 2.605 | 2.605 | 2.605 | 2.605 | 2.676 | 2.676 | 2.676 | 2.676 | 2.676 | 2.192 | 2.192 | 2.192 | 2.192 | 2.192 |

### Probe texts no model trained on

Bits per character on held-out prose (lower = the model finds it more natural).

| model | mccarthy_essays |
|---|---:|
| **mccarthy model** | 2.288 |
| **mccarthy_4ep model** | 2.548 |
| **mccarthy_d1 model** | 2.965 |
| **mccarthy_d2 model** | 2.671 |
| **mccarthy_d4 model** | 2.577 |
| **melville model** | 3.125 |
| **melville_4ep model** | 2.921 |
| **melville_d1 model** | 2.777 |
| **melville_d2 model** | 2.797 |
| **melville_d4 model** | 2.788 |
| **shakespeare model** | 3.231 |
| **shakespeare_4ep model** | 3.141 |
| **shakespeare_d1 model** | 3.198 |
| **shakespeare_d2 model** | 3.134 |
| **shakespeare_d4 model** | 3.151 |

## Attention-head layout (mean over held-out windows)

**mccarthy** — mean normalised entropy 0.531; mean attention distance 19.7 chars (layer means 4.3, 33.2, 8.6, 13.9, 22.5, 35.6); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H3 = 0.01; heads with induction > 0.1: 0.

**mccarthy_4ep** — mean normalised entropy 0.574; mean attention distance 21.0 chars (layer means 4.9, 25.1, 11.4, 16.0, 26.3, 42.3); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L5H1 = 0.01; heads with induction > 0.1: 0.

**mccarthy_d1** — mean normalised entropy 0.332; mean attention distance 9.4 chars (layer means 9.4); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L0H0 = 0.01; heads with induction > 0.1: 0.

**mccarthy_d2** — mean normalised entropy 0.433; mean attention distance 15.5 chars (layer means 9.1, 21.9); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L0H4 = 0.01; heads with induction > 0.1: 0.

**mccarthy_d4** — mean normalised entropy 0.530; mean attention distance 18.7 chars (layer means 5.5, 13.0, 22.0, 34.4); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L2H5 = 0.01; heads with induction > 0.1: 0.

**melville** — mean normalised entropy 0.477; mean attention distance 18.2 chars (layer means 5.0, 34.9, 5.6, 21.2, 17.8, 24.9); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L3H3 = 0.01; heads with induction > 0.1: 0.

**melville_4ep** — mean normalised entropy 0.532; mean attention distance 19.8 chars (layer means 5.2, 36.1, 9.4, 13.2, 21.3, 33.6); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H2 = 0.01; heads with induction > 0.1: 0.

**melville_d1** — mean normalised entropy 0.262; mean attention distance 9.7 chars (layer means 9.7); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L0H4 = 0.00; heads with induction > 0.1: 0.

**melville_d2** — mean normalised entropy 0.397; mean attention distance 13.1 chars (layer means 7.9, 18.3); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L0H5 = 0.00; heads with induction > 0.1: 0.

**melville_d4** — mean normalised entropy 0.473; mean attention distance 18.0 chars (layer means 5.4, 22.1, 13.8, 30.8); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L3H5 = 0.01; heads with induction > 0.1: 0.

**shakespeare** — mean normalised entropy 0.491; mean attention distance 21.4 chars (layer means 3.6, 50.0, 6.7, 9.4, 25.9, 32.5); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H4 = 0.07; heads with induction > 0.1: 0.

**shakespeare_4ep** — mean normalised entropy 0.564; mean attention distance 21.1 chars (layer means 5.5, 44.2, 7.9, 11.1, 22.0, 35.6); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H5 = 0.01; heads with induction > 0.1: 0.

**shakespeare_d1** — mean normalised entropy 0.295; mean attention distance 8.9 chars (layer means 8.9); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L0H3 = 0.00; heads with induction > 0.1: 0.

**shakespeare_d2** — mean normalised entropy 0.447; mean attention distance 14.6 chars (layer means 6.5, 22.6); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H4 = 0.00; heads with induction > 0.1: 0.

**shakespeare_d4** — mean normalised entropy 0.537; mean attention distance 18.8 chars (layer means 5.4, 29.2, 12.0, 28.8); 2 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H1 = 0.01; heads with induction > 0.1: 0.

## Spectra of the learned operators

Effective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).

| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 20.4 | 24.8 | 152.1 | 28.5 | -6.32 | -6.31 | -0.46 |
| mccarthy_4ep | 16.2 | 21.4 | 181.9 | 28.0 | -6.32 | -6.33 | -0.40 |
| mccarthy_d1 | 29.5 | 47.3 | 135.2 | 29.0 | -6.34 | -6.31 | -0.51 |
| mccarthy_d2 | 21.5 | 35.4 | 172.2 | 27.3 | -6.33 | -6.33 | -0.44 |
| mccarthy_d4 | 18.4 | 27.2 | 179.1 | 27.1 | -6.32 | -6.33 | -0.41 |
| melville | 20.5 | 26.0 | 147.1 | 33.5 | -6.32 | -6.31 | -0.47 |
| melville_4ep | 16.6 | 22.7 | 156.4 | 30.3 | -6.31 | -6.32 | -0.45 |
| melville_d1 | 26.2 | 43.4 | 115.8 | 36.7 | -6.34 | -6.31 | -0.57 |
| melville_d2 | 23.2 | 35.7 | 153.0 | 32.1 | -6.33 | -6.33 | -0.48 |
| melville_d4 | 18.1 | 26.5 | 159.0 | 31.5 | -6.32 | -6.32 | -0.46 |
| shakespeare | 20.9 | 27.3 | 147.6 | 30.8 | -6.33 | -6.32 | -0.47 |
| shakespeare_4ep | 17.2 | 23.7 | 184.6 | 34.4 | -6.31 | -6.33 | -0.40 |
| shakespeare_d1 | 26.1 | 49.9 | 149.8 | 36.1 | -6.34 | -6.30 | -0.48 |
| shakespeare_d2 | 20.5 | 38.6 | 180.9 | 34.7 | -6.33 | -6.32 | -0.42 |
| shakespeare_d4 | 18.7 | 28.8 | 184.4 | 35.5 | -6.32 | -6.33 | -0.40 |

## Martin–Mahoney alpha (power-law tail of the eigenvalue spectrum of WᵀW)

Near 2 = well-trained layer; below 2 = over-trained; above 6 = random. Mean over layers.

| model | Wq | Wk | Wv | Wo | mlp_in | mlp_out |
|---|---:|---:|---:|---:|---:|---:|
| mccarthy | 2.13 | 2.51 | 2.19 | 2.19 | 1.71 | 1.98 |
| mccarthy_4ep | 2.26 | 2.42 | 2.98 | 1.77 | 2.17 | 1.84 |
| mccarthy_d1 | 1.79 | 1.92 | 2.67 | 2.18 | 1.87 | 1.95 |
| mccarthy_d2 | 2.07 | 2.16 | 2.90 | 1.97 | 1.99 | 1.92 |
| mccarthy_d4 | 2.18 | 2.30 | 2.91 | 1.86 | 2.11 | 1.87 |
| melville | 2.08 | 2.32 | 2.20 | 2.41 | 1.71 | 1.99 |
| melville_4ep | 1.97 | 2.07 | 2.62 | 1.95 | 1.94 | 1.78 |
| melville_d1 | 1.72 | 1.78 | 2.23 | 1.81 | 1.70 | 1.83 |
| melville_d2 | 1.82 | 1.92 | 2.37 | 2.27 | 1.82 | 1.80 |
| melville_d4 | 1.94 | 2.03 | 2.55 | 1.98 | 1.91 | 1.80 |
| shakespeare | 2.12 | 2.31 | 1.84 | 2.07 | 1.70 | 2.13 |
| shakespeare_4ep | 2.31 | 2.45 | 2.90 | 1.75 | 2.16 | 1.80 |
| shakespeare_d1 | 1.88 | 1.97 | 2.70 | 1.98 | 1.84 | 1.88 |
| shakespeare_d2 | 2.08 | 2.19 | 2.87 | 2.01 | 2.01 | 1.88 |
| shakespeare_d4 | 2.22 | 2.34 | 2.92 | 1.83 | 2.12 | 1.84 |

## Logit lens: bits/char if the model stopped after block k

| model | embed | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 25.50 | 3.88 | 2.98 | 2.47 | 1.96 | 1.62 | 1.51 |
| mccarthy_4ep | 16.71 | 3.54 | 2.79 | 2.39 | 2.03 | 1.84 | 1.78 |
| mccarthy_d1 | 16.87 | 2.04 |
| mccarthy_d2 | 16.92 | 2.75 | 1.88 |
| mccarthy_d4 | 16.72 | 3.38 | 2.47 | 1.94 | 1.79 |
| melville | 24.65 | 4.16 | 3.44 | 3.02 | 2.59 | 2.16 | 1.92 |
| melville_4ep | 19.59 | 3.98 | 3.35 | 2.93 | 2.55 | 2.23 | 2.08 |
| melville_d1 | 19.38 | 2.44 |
| melville_d2 | 19.20 | 3.09 | 2.28 |
| melville_d4 | 19.61 | 3.76 | 2.98 | 2.39 | 2.15 |
| shakespeare | 25.20 | 3.98 | 3.32 | 2.91 | 2.47 | 2.04 | 1.90 |
| shakespeare_4ep | 16.78 | 3.85 | 3.31 | 2.88 | 2.51 | 2.25 | 2.17 |
| shakespeare_d1 | 17.43 | 2.40 |
| shakespeare_d2 | 16.93 | 3.03 | 2.29 |
| shakespeare_d4 | 16.86 | 3.58 | 2.86 | 2.37 | 2.20 |

## Residual stream and position

**mccarthy** — residual norm by block: 1.5, 13.1, 20.8, 28.1, 35.0, 42.9, 52.4; attention:MLP update ratio per block: 0.44, 0.62, 0.78, 0.67, 0.51, 0.36; position-embedding spectral centroid 29.8 cycles/window, 23% of power below 8 cycles.

**mccarthy_4ep** — residual norm by block: 1.0, 8.5, 12.0, 14.9, 18.5, 22.6, 26.7; attention:MLP update ratio per block: 0.37, 0.67, 0.76, 0.50, 0.37, 0.37; position-embedding spectral centroid 35.5 cycles/window, 19% of power below 8 cycles.

**mccarthy_d1** — residual norm by block: 1.0, 13.3; attention:MLP update ratio per block: 0.19; position-embedding spectral centroid 32.9 cycles/window, 23% of power below 8 cycles.

**mccarthy_d2** — residual norm by block: 1.0, 8.3, 17.6; attention:MLP update ratio per block: 0.39, 0.30; position-embedding spectral centroid 34.7 cycles/window, 23% of power below 8 cycles.

**mccarthy_d4** — residual norm by block: 1.0, 8.0, 12.2, 16.9, 22.1; attention:MLP update ratio per block: 0.37, 0.67, 0.42, 0.31; position-embedding spectral centroid 33.6 cycles/window, 24% of power below 8 cycles.

**melville** — residual norm by block: 1.5, 13.1, 21.2, 28.7, 34.9, 42.2, 52.4; attention:MLP update ratio per block: 0.37, 0.64, 0.78, 0.69, 0.59, 0.41; position-embedding spectral centroid 30.7 cycles/window, 22% of power below 8 cycles.

**melville_4ep** — residual norm by block: 1.1, 10.2, 14.8, 18.8, 22.8, 27.8, 32.5; attention:MLP update ratio per block: 0.32, 0.75, 0.84, 0.65, 0.43, 0.45; position-embedding spectral centroid 34.1 cycles/window, 19% of power below 8 cycles.

**melville_d1** — residual norm by block: 1.1, 17.0; attention:MLP update ratio per block: 0.16; position-embedding spectral centroid 30.3 cycles/window, 23% of power below 8 cycles.

**melville_d2** — residual norm by block: 1.1, 10.0, 22.7; attention:MLP update ratio per block: 0.31, 0.30; position-embedding spectral centroid 30.3 cycles/window, 25% of power below 8 cycles.

**melville_d4** — residual norm by block: 1.1, 9.5, 14.6, 20.9, 26.5; attention:MLP update ratio per block: 0.35, 0.67, 0.51, 0.42; position-embedding spectral centroid 31.3 cycles/window, 25% of power below 8 cycles.

**shakespeare** — residual norm by block: 1.5, 12.9, 19.6, 26.9, 34.5, 42.3, 52.5; attention:MLP update ratio per block: 0.38, 0.59, 0.82, 0.69, 0.58, 0.41; position-embedding spectral centroid 32.5 cycles/window, 24% of power below 8 cycles.

**shakespeare_4ep** — residual norm by block: 1.0, 8.1, 10.4, 13.3, 16.7, 20.7, 25.3; attention:MLP update ratio per block: 0.32, 0.64, 0.96, 0.60, 0.42, 0.40; position-embedding spectral centroid 36.4 cycles/window, 17% of power below 8 cycles.

**shakespeare_d1** — residual norm by block: 1.0, 12.1; attention:MLP update ratio per block: 0.19; position-embedding spectral centroid 32.3 cycles/window, 23% of power below 8 cycles.

**shakespeare_d2** — residual norm by block: 1.0, 7.3, 16.1; attention:MLP update ratio per block: 0.35, 0.26; position-embedding spectral centroid 33.5 cycles/window, 24% of power below 8 cycles.

**shakespeare_d4** — residual norm by block: 1.0, 7.4, 10.5, 15.3, 20.3; attention:MLP update ratio per block: 0.36, 0.56, 0.49, 0.37; position-embedding spectral centroid 33.4 cycles/window, 21% of power below 8 cycles.

## Worldview probe: the first word each model puts after a shared prompt

96 sampled continuations per prompt (temperature 0.9, top-k 40); counts of the first word.

## Samples (temperature 0.8, top-k 40)

### mccarthy

### mccarthy_4ep

### mccarthy_d1

### mccarthy_d2

### mccarthy_d4

### melville

### melville_4ep

### melville_d1

### melville_d2

### melville_d4

### shakespeare

### shakespeare_4ep

### shakespeare_d1

### shakespeare_d2

### shakespeare_d4


## Figures

- `fig_cross_bpc.png`
- `fig_cross_bpc_sweep.png`
- `fig_delta.png`
- `fig_eff_rank.png`
- `fig_eff_rank_sweep.png`
- `fig_head_layout.png`
- `fig_head_layout_sweep.png`
- `fig_induction.png`
- `fig_induction_sweep.png`
- `fig_ncd.png`
- `fig_resid_positional.png`
- `fig_resid_positional_sweep.png`
- `fig_spectra.png`
- `fig_spectra_sweep.png`