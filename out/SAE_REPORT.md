# Sparse autoencoders on the 2-layer persona models

Site: residual stream after block 1 of 2 (d=384). Dictionary 8×. One per author, same recipe.

| model | l1 | var. explained | L0 (features/position) | dead | live |
|---|---:|---:|---:|---:|---:|
| mccarthy_d2 | 3.0 | 0.896 | 10.5 | 1% | 3026 |
| melville_d2 | 3.0 | 0.895 | 10.1 | 1% | 3056 |
| shakespeare_d2 | 3.0 | 0.883 | 11.4 | 1% | 3033 |
| shakespeare_noheads_d2 | 3.0 | 0.890 | 10.7 | 1% | 3030 |
| shakespeare_nonum_d2 | 3.0 | 0.884 | 11.4 | 1% | 3028 |

## Feature-type mix (share of live features)

Typed by what a feature fires on (input side) and the character its decoder direction promotes through the unembedding (output side).

| type | mccarthy_d2 | melville_d2 | shakespeare_d2 | shakespeare_noheads_d2 | shakespeare_nonum_d2 |
|---|---:|---:|---:|---:|---:|
| after-char | 16.9% | 13.2% | 15.9% | 14.4% | 15.8% |
| predicts capital | 1.8% | 0.4% | 46.2% | 9.7% | 42.5% |
| predicts consonant | 52.3% | 44.1% | 13.2% | 35.3% | 18.2% |
| predicts punctuation | 1.3% | 3.3% | 12.1% | 8.4% | 14.0% |
| predicts vowel | 14.3% | 32.9% | 7.0% | 26.8% | 4.8% |
| predicts word-end | 13.4% | 6.2% | 5.6% | 5.3% | 4.8% |

## Density histogram (log10 firing rate, bins from 1e-6 to 1)

| model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mccarthy_d2 | 0 | 0 | 360 | 730 | 502 | 231 | 232 | 695 | 262 | 12 | 0 | 2 |
| melville_d2 | 0 | 0 | 260 | 608 | 539 | 289 | 302 | 859 | 187 | 8 | 2 | 2 |
| shakespeare_d2 | 0 | 0 | 356 | 704 | 514 | 237 | 226 | 671 | 309 | 14 | 1 | 1 |
| shakespeare_noheads_d2 | 0 | 0 | 434 | 861 | 490 | 178 | 171 | 568 | 316 | 9 | 2 | 1 |
| shakespeare_nonum_d2 | 0 | 0 | 372 | 652 | 509 | 278 | 233 | 637 | 335 | 8 | 2 | 2 |

## Most frequent live features

### mccarthy_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2902 | 0.385 | `' '` (H=3.05) | 'l' ' ' 's' | predicts consonant |
| 2241 | 0.368 | `' '` (H=3.05) | 'l' 'e' ' ' | predicts consonant |
| 1087 | 0.080 | `' '` (H=0.18) | 'a' 's' 't' | after-char |
| 1134 | 0.068 | `' '` (H=0.26) | 'a' 'o' 'd' | after-char |
| 2013 | 0.060 | `'n'` (H=2.84) | ' ' 'e' '.' | predicts word-end |
| 2741 | 0.043 | `' '` (H=2.95) | 'e' ' ' 'l' | predicts vowel |
| 373 | 0.040 | `' '` (H=2.82) | 'o' ' ' 'd' | predicts vowel |
| 1716 | 0.038 | `'o'` (H=2.62) | 'i' 'l' 'e' | predicts vowel |
| 2695 | 0.036 | `' '` (H=1.53) | 'I' 'd' 'c' | predicts capital |
| 1261 | 0.036 | `' '` (H=0.31) | 'l' 'c' 's' | after-char |
| 2558 | 0.034 | `' '` (H=0.42) | 'a' 's' 't' | after-char |
| 2453 | 0.033 | `' '` (H=2.9) | 'l' 's' ' ' | predicts consonant |
| 2367 | 0.033 | `'t'` (H=0.42) | 'h' 'r' 'o' | after-char |
| 1345 | 0.032 | `'t'` (H=3.06) | 'o' 'e' 'u' | predicts vowel |
| 1373 | 0.031 | `' '` (H=2.73) | 's' 'd' 'g' | predicts consonant |

### melville_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1113 | 0.336 | `'e'` (H=3.22) | 'e' 'n' 'l' | predicts vowel |
| 2088 | 0.329 | `'e'` (H=3.2) | 'e' 'n' 's' | predicts vowel |
| 3027 | 0.102 | `' '` (H=0.2) | 'a' 't' 'c' | after-char |
| 595 | 0.101 | `'e'` (H=3.27) | '\n' 'e' 'n' | predicts word-end |
| 898 | 0.093 | `' '` (H=0.45) | 't' 'a' 's' | after-char |
| 2581 | 0.089 | `' '` (H=0.44) | 't' 'a' 's' | after-char |
| 1087 | 0.081 | `' '` (H=0.34) | 'a' 't' 'o' | after-char |
| 214 | 0.075 | `'t'` (H=3.09) | 'e' 'a' 'o' | predicts vowel |
| 309 | 0.055 | `' '` (H=3.36) | 'a' 'e' 'r' | predicts vowel |
| 2902 | 0.052 | `'t'` (H=2.95) | 'o' 'e' 'a' | predicts vowel |
| 2009 | 0.033 | `' '` (H=0.27) | 's' 'a' 't' | after-char |
| 1629 | 0.033 | `' '` (H=0.2) | 't' 's' 'a' | after-char |
| 2617 | 0.032 | `'t'` (H=0.54) | 'h' 'o' 'r' | after-char |
| 2941 | 0.029 | `'t'` (H=3.13) | 'e' 'n' 'o' | predicts vowel |
| 1211 | 0.028 | `' '` (H=0.29) | 's' 'l' 'c' | after-char |

### shakespeare_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2453 | 0.365 | `' '` (H=3.3) | 'E' '.' 'T' | predicts capital |
| 2442 | 0.311 | `' '` (H=3.33) | 'T' ' ' 'e' | predicts capital |
| 2922 | 0.095 | `' '` (H=0.71) | 'T' 't' 'I' | predicts capital |
| 2745 | 0.094 | `'e'` (H=2.99) | ' ' 'd' 'T' | predicts word-end |
| 656 | 0.091 | `' '` (H=0.37) | 't' 'a' 'o' | after-char |
| 1716 | 0.091 | `'t'` (H=3.28) | 'o' 'e' 'u' | predicts vowel |
| 1087 | 0.077 | `' '` (H=0.21) | 't' 'o' 'a' | after-char |
| 1512 | 0.054 | `' '` (H=0.35) | 'I' 'i' 'd' | after-char |
| 1971 | 0.054 | `'t'` (H=3.09) | 'o' 'i' 'u' | predicts vowel |
| 2109 | 0.048 | `'r'` (H=2.71) | '.' ',' 's' | predicts punctuation |
| 476 | 0.047 | `'t'` (H=3.02) | 'o' 'u' 'l' | predicts vowel |
| 1219 | 0.044 | `' '` (H=0.4) | 't' 'a' 'I' | after-char |
| 3027 | 0.035 | `' '` (H=0.27) | 'a' 't' 'd' | after-char |
| 1784 | 0.034 | `' '` (H=0.79) | 'o' 'T' 't' | predicts vowel |
| 360 | 0.034 | `'h'` (H=2.74) | 'a' 'r' 'o' | predicts vowel |

### shakespeare_noheads_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 476 | 0.344 | `' '` (H=3.24) | 'e' ' ' '’' | predicts vowel |
| 1977 | 0.302 | `'e'` (H=3.27) | 'e' 'r' 'n' | predicts vowel |
| 1087 | 0.101 | `' '` (H=0.29) | 't' 'a' 's' | after-char |
| 1866 | 0.096 | `' '` (H=0.38) | 't' 's' 'a' | after-char |
| 1784 | 0.092 | `' '` (H=0.21) | 't' 'o' 'a' | after-char |
| 1113 | 0.082 | `'t'` (H=2.74) | '.' ',' 'e' | predicts punctuation |
| 1292 | 0.063 | `'s'` (H=2.8) | 'e' ' ' ',' | predicts vowel |
| 1159 | 0.061 | `'s'` (H=3.1) | 'e' 'o' 'u' | predicts vowel |
| 353 | 0.037 | `' '` (H=0.23) | 't' 'e' 'a' | after-char |
| 379 | 0.037 | `'l'` (H=2.9) | 'e' 'u' 'o' | predicts vowel |
| 338 | 0.034 | `' '` (H=0.4) | 't' 's' 'e' | after-char |
| 447 | 0.034 | `'t'` (H=3.18) | 'e' 'o' 'u' | predicts vowel |
| 1599 | 0.032 | `' '` (H=0.4) | 'd' 'e' 's' | after-char |
| 2617 | 0.031 | `'t'` (H=1.01) | 'h' 'r' 'o' | predicts consonant |
| 1150 | 0.031 | `'t'` (H=3.2) | 'e' 'r' 'n' | predicts vowel |

### shakespeare_nonum_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1270 | 0.352 | `' '` (H=3.3) | 'E' '.' 'O' | predicts capital |
| 2745 | 0.321 | `' '` (H=3.31) | 's' 'n' ' ' | predicts consonant |
| 2453 | 0.163 | `'e'` (H=3.22) | '.' ',' '\n' | predicts punctuation |
| 2661 | 0.107 | `'e'` (H=3.2) | '.' ',' ' ' | predicts punctuation |
| 1784 | 0.091 | `' '` (H=0.26) | 'o' 't' 'a' | after-char |
| 1087 | 0.089 | `' '` (H=0.42) | 't' 'I' 's' | after-char |
| 373 | 0.059 | `'h'` (H=3.33) | 'o' 'a' 'i' | predicts vowel |
| 344 | 0.033 | `' '` (H=3.02) | '.' ',' 's' | predicts punctuation |
| 329 | 0.033 | `' '` (H=0.34) | 'I' 't' 's' | after-char |
| 270 | 0.033 | `'e'` (H=3.26) | '.' 'e' 'R' | predicts punctuation |
| 2432 | 0.032 | `'r'` (H=2.87) | '.' ',' 's' | predicts punctuation |
| 2459 | 0.032 | `'h'` (H=3.04) | 'a' 'u' 'i' | predicts vowel |
| 2059 | 0.031 | `'e'` (H=3.23) | '.' 't' 'l' | predicts punctuation |
| 2617 | 0.031 | `'t'` (H=0.76) | 'h' 'r' 'o' | predicts consonant |
| 2050 | 0.031 | `'e'` (H=3.14) | '.' ',' 's' | predicts punctuation |
