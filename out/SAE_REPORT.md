# Sparse autoencoders on the 2-layer persona models

Site: residual stream after block 1 of 2 (d=384). Dictionary 8×. One per author, same recipe.

| model | l1 | var. explained | L0 (features/position) | dead | live |
|---|---:|---:|---:|---:|---:|
| mccarthy_d2 | 3.0 | 0.896 | 10.8 | 1% | 3031 |
| melville_d2 | 3.0 | 0.895 | 9.9 | 1% | 3045 |
| shakespeare_d2 | 3.0 | 0.883 | 11.5 | 1% | 3029 |
| shakespeare_nonum_d2 | 3.0 | 0.882 | 11.7 | 1% | 3029 |

## Feature-type mix (share of live features)

Typed by what a feature fires on (input side) and the character its decoder direction promotes through the unembedding (output side).

| type | mccarthy_d2 | melville_d2 | shakespeare_d2 | shakespeare_nonum_d2 |
|---|---:|---:|---:|---:|
| after-char | 19.4% | 11.7% | 14.0% | 14.3% |
| predicts capital | 1.8% | 0.4% | 48.2% | 43.5% |
| predicts consonant | 50.6% | 44.9% | 13.2% | 18.5% |
| predicts punctuation | 1.3% | 3.3% | 12.1% | 14.1% |
| predicts vowel | 13.7% | 33.6% | 7.1% | 5.0% |
| predicts word-end | 13.2% | 6.1% | 5.4% | 4.7% |

## Density histogram (log10 firing rate, bins from 1e-6 to 1)

| model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mccarthy_d2 | 0 | 0 | 440 | 693 | 475 | 224 | 230 | 684 | 271 | 12 | 0 | 2 |
| melville_d2 | 0 | 0 | 217 | 626 | 548 | 295 | 291 | 871 | 185 | 9 | 1 | 2 |
| shakespeare_d2 | 0 | 0 | 339 | 686 | 550 | 231 | 227 | 671 | 309 | 14 | 1 | 1 |
| shakespeare_nonum_d2 | 0 | 0 | 371 | 636 | 533 | 266 | 233 | 640 | 338 | 8 | 2 | 2 |

## Most frequent live features

### mccarthy_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2902 | 0.386 | `' '` (H=3.06) | 'l' ' ' 's' | predicts consonant |
| 2241 | 0.368 | `' '` (H=3.03) | 'l' 'e' ' ' | predicts consonant |
| 1087 | 0.079 | `' '` (H=0.18) | 'a' 's' 't' | after-char |
| 1134 | 0.068 | `' '` (H=0.27) | 'a' 'o' 'd' | after-char |
| 2013 | 0.059 | `'n'` (H=2.85) | ' ' 'e' '.' | predicts word-end |
| 2741 | 0.043 | `' '` (H=2.96) | 'e' ' ' 'l' | predicts vowel |
| 373 | 0.041 | `' '` (H=2.79) | 'o' ' ' 'd' | predicts vowel |
| 1716 | 0.038 | `'o'` (H=2.63) | 'i' 'l' 'e' | predicts vowel |
| 1261 | 0.035 | `' '` (H=0.35) | 'l' 'c' 's' | after-char |
| 2695 | 0.035 | `' '` (H=1.52) | 'I' 'd' 'c' | predicts capital |
| 2558 | 0.034 | `' '` (H=0.38) | 'a' 's' 't' | after-char |
| 2453 | 0.033 | `' '` (H=2.92) | 'l' 's' ' ' | predicts consonant |
| 1345 | 0.033 | `'h'` (H=3.02) | 'o' 'e' 'u' | predicts vowel |
| 2367 | 0.032 | `'t'` (H=0.45) | 'h' 'r' 'o' | after-char |
| 1373 | 0.031 | `' '` (H=2.76) | 's' 'd' 'g' | predicts consonant |

### melville_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1113 | 0.337 | `'e'` (H=3.25) | 'e' 'n' 'l' | predicts vowel |
| 2088 | 0.329 | `'e'` (H=3.23) | 'e' 'n' 's' | predicts vowel |
| 3027 | 0.103 | `' '` (H=0.18) | 'a' 't' 'c' | after-char |
| 595 | 0.098 | `'e'` (H=3.32) | '\n' 'e' 'n' | predicts word-end |
| 898 | 0.093 | `' '` (H=0.44) | 't' 'a' 's' | after-char |
| 2581 | 0.090 | `' '` (H=0.44) | 't' 'a' 's' | after-char |
| 1087 | 0.081 | `' '` (H=0.34) | 'a' 't' 'o' | after-char |
| 214 | 0.074 | `'t'` (H=3.12) | 'e' 'a' 'o' | predicts vowel |
| 309 | 0.057 | `' '` (H=3.43) | 'a' 'e' 'r' | predicts vowel |
| 2902 | 0.052 | `'t'` (H=2.94) | 'o' 'e' 'a' | predicts vowel |
| 2009 | 0.033 | `' '` (H=0.32) | 's' 'a' 't' | after-char |
| 1629 | 0.033 | `' '` (H=0.19) | 't' 's' 'a' | after-char |
| 2617 | 0.031 | `'t'` (H=0.53) | 'h' 'o' 'r' | after-char |
| 2941 | 0.028 | `'t'` (H=3.13) | 'e' 'n' 'o' | predicts vowel |
| 1211 | 0.028 | `' '` (H=0.31) | 's' 'l' 'c' | after-char |

### shakespeare_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2453 | 0.365 | `' '` (H=3.3) | 'E' '.' 'T' | predicts capital |
| 2442 | 0.312 | `' '` (H=3.34) | 'T' ' ' 'e' | predicts capital |
| 2922 | 0.094 | `' '` (H=0.71) | 'T' 't' 'I' | predicts capital |
| 2745 | 0.093 | `'e'` (H=3.03) | ' ' 'd' 'T' | predicts word-end |
| 656 | 0.091 | `' '` (H=0.35) | 't' 'a' 'o' | after-char |
| 1716 | 0.090 | `'t'` (H=3.28) | 'o' 'e' 'u' | predicts vowel |
| 1087 | 0.077 | `' '` (H=0.21) | 't' 'o' 'a' | after-char |
| 1971 | 0.055 | `'t'` (H=3.08) | 'o' 'i' 'u' | predicts vowel |
| 1512 | 0.054 | `' '` (H=0.28) | 'I' 'i' 'd' | after-char |
| 476 | 0.048 | `'t'` (H=3.06) | 'o' 'u' 'l' | predicts vowel |
| 2109 | 0.048 | `'r'` (H=2.71) | '.' ',' 's' | predicts punctuation |
| 1219 | 0.044 | `' '` (H=0.38) | 't' 'a' 'I' | after-char |
| 3027 | 0.036 | `' '` (H=0.22) | 'a' 't' 'd' | after-char |
| 360 | 0.034 | `'h'` (H=2.69) | 'a' 'r' 'o' | predicts vowel |
| 1361 | 0.034 | `'e'` (H=0.48) | '.' ',' 'd' | after-char |

### shakespeare_nonum_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1270 | 0.352 | `' '` (H=3.3) | 'E' '.' 'O' | predicts capital |
| 2745 | 0.321 | `' '` (H=3.31) | 's' 'n' ' ' | predicts consonant |
| 2453 | 0.166 | `'e'` (H=3.22) | '.' ',' '\n' | predicts punctuation |
| 2661 | 0.106 | `'e'` (H=3.2) | '.' ',' ' ' | predicts punctuation |
| 1784 | 0.089 | `' '` (H=0.27) | 'o' 't' 'a' | after-char |
| 1087 | 0.089 | `' '` (H=0.39) | 't' 'I' 's' | after-char |
| 373 | 0.060 | `'h'` (H=3.34) | 'o' 'a' 'i' | predicts vowel |
| 2432 | 0.034 | `'r'` (H=2.88) | '.' ',' 's' | predicts punctuation |
| 270 | 0.033 | `'e'` (H=3.23) | '.' 'e' 'R' | predicts punctuation |
| 344 | 0.032 | `' '` (H=3.0) | '.' ',' 's' | predicts punctuation |
| 329 | 0.032 | `' '` (H=0.34) | 'I' 't' 's' | after-char |
| 2459 | 0.032 | `'h'` (H=3.03) | 'a' 'u' 'i' | predicts vowel |
| 2356 | 0.032 | `','` (H=1.65) | '\n' ' ' '.' | predicts word-end |
| 2059 | 0.031 | `'e'` (H=3.2) | '.' 't' 'l' | predicts punctuation |
| 2972 | 0.030 | `','` (H=2.27) | ' ' '\n' 's' | predicts word-end |
