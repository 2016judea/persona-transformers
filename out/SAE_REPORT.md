# Sparse autoencoders on the 2-layer persona models

Site: residual stream after block 1 of 2 (d=384). Dictionary 8×. One per author, same recipe.

| model | l1 | var. explained | L0 (features/position) | dead | live |
|---|---:|---:|---:|---:|---:|
| mccarthy_d2 | 3.0 | 0.896 | 10.4 | 2% | 3018 |
| melville_d2 | 3.0 | 0.893 | 9.9 | 1% | 3049 |
| shakespeare_d2 | 3.0 | 0.882 | 11.4 | 1% | 3046 |

## Feature-type mix (share of live features)

Typed by what a feature fires on (input side) and the character its decoder direction promotes through the unembedding (output side).

| type | mccarthy_d2 | melville_d2 | shakespeare_d2 |
|---|---:|---:|---:|
| after-char | 19.2% | 14.4% | 13.7% |
| predicts capital | 1.7% | 0.4% | 48.2% |
| predicts consonant | 50.5% | 43.3% | 13.2% |
| predicts punctuation | 1.3% | 3.1% | 12.3% |
| predicts vowel | 13.9% | 32.8% | 7.2% |
| predicts word-end | 13.3% | 5.9% | 5.4% |

## Density histogram (log10 firing rate, bins from 1e-6 to 1)

| model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mccarthy_d2 | 0 | 0 | 458 | 694 | 437 | 238 | 220 | 692 | 265 | 12 | 0 | 2 |
| melville_d2 | 0 | 0 | 285 | 618 | 507 | 289 | 291 | 860 | 187 | 8 | 2 | 2 |
| shakespeare_d2 | 0 | 0 | 336 | 663 | 579 | 241 | 237 | 664 | 309 | 15 | 1 | 1 |

## Most frequent live features

### mccarthy_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2902 | 0.385 | `' '` (H=3.03) | 'l' ' ' 's' | predicts consonant |
| 2241 | 0.368 | `' '` (H=3.03) | 'l' 'e' ' ' | predicts consonant |
| 1087 | 0.079 | `' '` (H=0.21) | 'a' 's' 't' | after-char |
| 1134 | 0.068 | `' '` (H=0.25) | 'a' 'o' 'd' | after-char |
| 2013 | 0.058 | `'n'` (H=2.84) | ' ' 'e' '.' | predicts word-end |
| 2741 | 0.042 | `' '` (H=2.94) | 'e' ' ' 'l' | predicts vowel |
| 373 | 0.039 | `' '` (H=2.81) | 'o' ' ' 'd' | predicts vowel |
| 1716 | 0.039 | `'o'` (H=2.61) | 'i' 'l' 'e' | predicts vowel |
| 2695 | 0.035 | `' '` (H=1.54) | 'I' 'd' 'c' | predicts capital |
| 1261 | 0.035 | `' '` (H=0.31) | 'l' 'c' 's' | after-char |
| 2558 | 0.034 | `' '` (H=0.37) | 'a' 's' 't' | after-char |
| 2367 | 0.034 | `'t'` (H=0.43) | 'h' 'r' 'o' | after-char |
| 2453 | 0.033 | `' '` (H=2.89) | 'l' 's' ' ' | predicts consonant |
| 1345 | 0.032 | `'h'` (H=3.01) | 'o' 'e' 'u' | predicts vowel |
| 1373 | 0.031 | `' '` (H=2.76) | 's' 'd' 'g' | predicts consonant |

### melville_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1113 | 0.337 | `'e'` (H=3.24) | 'e' 'n' 'l' | predicts vowel |
| 2088 | 0.327 | `'e'` (H=3.17) | 'e' 'n' 's' | predicts vowel |
| 595 | 0.103 | `'e'` (H=3.29) | '\n' 'e' 'n' | predicts word-end |
| 3027 | 0.102 | `' '` (H=0.18) | 'a' 't' 'c' | after-char |
| 898 | 0.093 | `' '` (H=0.46) | 't' 'a' 's' | after-char |
| 2581 | 0.089 | `' '` (H=0.45) | 't' 'a' 's' | after-char |
| 1087 | 0.080 | `' '` (H=0.33) | 'a' 't' 'o' | after-char |
| 214 | 0.075 | `'t'` (H=3.07) | 'e' 'a' 'o' | predicts vowel |
| 309 | 0.055 | `' '` (H=3.33) | 'a' 'e' 'r' | predicts vowel |
| 2902 | 0.052 | `'t'` (H=2.96) | 'o' 'e' 'a' | predicts vowel |
| 2009 | 0.032 | `' '` (H=0.29) | 's' 'a' 't' | after-char |
| 1629 | 0.032 | `' '` (H=0.18) | 't' 's' 'a' | after-char |
| 2617 | 0.031 | `'t'` (H=0.54) | 'h' 'o' 'r' | after-char |
| 2941 | 0.028 | `'t'` (H=3.11) | 'e' 'n' 'o' | predicts vowel |
| 674 | 0.028 | `'e'` (H=2.89) | 'e' 'l' 's' | predicts vowel |

### shakespeare_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2453 | 0.366 | `' '` (H=3.28) | 'E' '.' 'T' | predicts capital |
| 2442 | 0.310 | `' '` (H=3.32) | 'T' ' ' 'e' | predicts capital |
| 2922 | 0.095 | `' '` (H=0.7) | 'T' 't' 'I' | after-char |
| 2745 | 0.093 | `'e'` (H=3.03) | ' ' 'd' 'T' | predicts word-end |
| 656 | 0.091 | `' '` (H=0.33) | 't' 'a' 'o' | after-char |
| 1716 | 0.090 | `'t'` (H=3.25) | 'o' 'e' 'u' | predicts vowel |
| 1087 | 0.077 | `' '` (H=0.21) | 't' 'o' 'a' | after-char |
| 1971 | 0.054 | `'t'` (H=3.09) | 'o' 'i' 'u' | predicts vowel |
| 1512 | 0.053 | `' '` (H=0.33) | 'I' 'i' 'd' | after-char |
| 2109 | 0.048 | `'r'` (H=2.64) | '.' ',' 's' | predicts punctuation |
| 476 | 0.047 | `'t'` (H=3.03) | 'o' 'u' 'l' | predicts vowel |
| 1219 | 0.042 | `' '` (H=0.39) | 't' 'a' 'I' | after-char |
| 3027 | 0.036 | `' '` (H=0.22) | 'a' 't' 'd' | after-char |
| 1784 | 0.034 | `' '` (H=0.78) | 'o' 'T' 't' | predicts vowel |
| 360 | 0.034 | `'h'` (H=2.67) | 'a' 'r' 'o' | predicts vowel |
