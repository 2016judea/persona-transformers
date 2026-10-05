# Sparse autoencoders on the 2-layer persona models

Site: residual stream after the block in the site column (d=384). Dictionary 8×. One per author, same recipe.

| model | site | l1 | var. explained | L0 (features/position) | dead | live |
|---|---:|---:|---:|---:|---:|---:|
| mccarthy_d2 | 1 | 3.0 | 0.895 | 10.7 | 2% | 3017 |
| mccarthy_d2_s2 | 2 | 3.0 | 0.880 | 10.3 | 0% | 3070 |
| melville_d2 | 1 | 3.0 | 0.894 | 10.1 | 1% | 3041 |
| melville_d2_s2 | 2 | 3.0 | 0.864 | 9.3 | 0% | 3066 |
| shakespeare_d2 | 1 | 3.0 | 0.878 | 11.8 | 1% | 3035 |
| shakespeare_d2_s2 | 2 | 3.0 | 0.877 | 9.2 | 0% | 3063 |
| shakespeare_noheads_d2 | 1 | 3.0 | 0.893 | 10.5 | 1% | 3026 |
| shakespeare_nonum_d2 | 1 | 3.0 | 0.881 | 11.2 | 1% | 3028 |

## Feature-type mix (share of live features)

Typed by what a feature fires on (input side) and the character its decoder direction promotes through the unembedding (output side).

| type | mccarthy_d2 | mccarthy_d2_s2 | melville_d2 | melville_d2_s2 | shakespeare_d2 | shakespeare_d2_s2 | shakespeare_noheads_d2 | shakespeare_nonum_d2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| after-char | 17.6% | 1.8% | 14.1% | 3.4% | 14.3% | 3.4% | 17.0% | 14.7% |
| predicts capital | 1.9% | 8.1% | 0.4% | 10.4% | 48.0% | 92.9% | 9.6% | 43.0% |
| predicts consonant | 51.1% | 30.5% | 43.8% | 84.8% | 13.2% | 1.9% | 34.6% | 18.4% |
| predicts punctuation | 1.3% | 59.0% | 3.4% | 0.8% | 12.0% | 1.7% | 8.2% | 14.1% |
| predicts vowel | 14.5% | 0.1% | 32.6% | 0.1% | 7.1% | 0.0% | 25.6% | 5.0% |
| predicts word-end | 13.8% | 0.6% | 5.8% | 0.5% | 5.4% | 0.2% | 5.1% | 4.8% |

## Density histogram (log10 firing rate, bins from 1e-6 to 1)

| model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mccarthy_d2 | 0 | 0 | 390 | 732 | 462 | 240 | 225 | 684 | 270 | 12 | 0 | 2 |
| mccarthy_d2_s2 | 0 | 0 | 42 | 539 | 1877 | 230 | 55 | 46 | 183 | 89 | 9 | 0 |
| melville_d2 | 0 | 0 | 281 | 621 | 513 | 274 | 311 | 842 | 187 | 9 | 1 | 2 |
| melville_d2_s2 | 0 | 0 | 109 | 776 | 1667 | 153 | 36 | 85 | 157 | 72 | 11 | 0 |
| shakespeare_d2 | 0 | 0 | 333 | 679 | 558 | 240 | 223 | 677 | 308 | 15 | 1 | 1 |
| shakespeare_d2_s2 | 0 | 0 | 109 | 647 | 1666 | 318 | 45 | 31 | 161 | 73 | 13 | 0 |
| shakespeare_noheads_d2 | 0 | 0 | 493 | 821 | 471 | 173 | 172 | 557 | 327 | 9 | 2 | 1 |
| shakespeare_nonum_d2 | 0 | 0 | 383 | 677 | 485 | 260 | 239 | 643 | 329 | 8 | 2 | 2 |

## Most frequent live features

### mccarthy_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2902 | 0.385 | `' '` (H=3.02) | 'l' ' ' 's' | predicts consonant |
| 2241 | 0.369 | `' '` (H=3.02) | 'l' 'e' ' ' | predicts consonant |
| 1087 | 0.080 | `' '` (H=0.17) | 'a' 's' 't' | after-char |
| 1134 | 0.069 | `' '` (H=0.23) | 'a' 'o' 'd' | after-char |
| 2013 | 0.059 | `'n'` (H=2.86) | ' ' 'e' '.' | predicts word-end |
| 2741 | 0.042 | `' '` (H=2.96) | 'e' ' ' 'l' | predicts vowel |
| 373 | 0.041 | `' '` (H=2.79) | 'o' ' ' 'd' | predicts vowel |
| 1716 | 0.039 | `'o'` (H=2.64) | 'i' 'l' 'e' | predicts vowel |
| 1261 | 0.035 | `' '` (H=0.31) | 'l' 'c' 's' | after-char |
| 2453 | 0.034 | `' '` (H=2.87) | 'l' 's' ' ' | predicts consonant |
| 2558 | 0.033 | `' '` (H=0.38) | 'a' 's' 't' | after-char |
| 2367 | 0.033 | `'t'` (H=0.39) | 'h' 'r' 'o' | after-char |
| 2695 | 0.033 | `' '` (H=1.54) | 'I' 'd' 'c' | predicts capital |
| 1345 | 0.033 | `'h'` (H=3.02) | 'o' 'e' 'u' | predicts vowel |
| 1373 | 0.032 | `' '` (H=2.74) | 's' 'd' 'g' | predicts consonant |

### mccarthy_d2_s2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 3039 | 0.205 | `' '` (H=3.08) | '_' 'Y' '0' | predicts punctuation |
| 3007 | 0.153 | `'e'` (H=2.79) | '?' '.' ',' | predicts punctuation |
| 194 | 0.135 | `' '` (H=0.56) | 'I' 'D' 'S' | after-char |
| 1433 | 0.134 | `' '` (H=0.46) | 'K' 'B' 'S' | after-char |
| 1257 | 0.131 | `' '` (H=0.5) | 'S' 'B' 'D' | after-char |
| 1261 | 0.125 | `' '` (H=0.71) | 'I' 'Y' 'D' | predicts capital |
| 173 | 0.124 | `'e'` (H=2.79) | ' ' ':' '?' | predicts word-end |
| 1318 | 0.111 | `'h'` (H=2.78) | 'I' '_' 'Y' | predicts capital |
| 1980 | 0.109 | `'d'` (H=2.78) | '.' '\n' ',' | predicts punctuation |
| 464 | 0.092 | `' '` (H=0.33) | 'H' 'D' 'Y' | after-char |
| 719 | 0.089 | `'h'` (H=2.5) | 'Y' '_' 'A' | predicts capital |
| 198 | 0.088 | `'r'` (H=2.92) | 'e' '_' 'E' | predicts vowel |
| 2894 | 0.083 | `'h'` (H=2.76) | '_' 'r' 'l' | predicts punctuation |
| 684 | 0.082 | `' '` (H=2.63) | 'F' 'D' 'T' | predicts capital |
| 1527 | 0.079 | `'e'` (H=2.67) | '.' ',' ':' | predicts punctuation |

### melville_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1113 | 0.337 | `'e'` (H=3.22) | 'e' 'n' 'l' | predicts vowel |
| 2088 | 0.328 | `'e'` (H=3.17) | 'e' 'n' 's' | predicts vowel |
| 3027 | 0.102 | `' '` (H=0.19) | 'a' 't' 'c' | after-char |
| 595 | 0.098 | `'e'` (H=3.27) | '\n' 'e' 'n' | predicts word-end |
| 898 | 0.093 | `' '` (H=0.44) | 't' 'a' 's' | after-char |
| 2581 | 0.089 | `' '` (H=0.44) | 't' 'a' 's' | after-char |
| 1087 | 0.081 | `' '` (H=0.34) | 'a' 't' 'o' | after-char |
| 214 | 0.075 | `'t'` (H=3.07) | 'e' 'a' 'o' | predicts vowel |
| 309 | 0.054 | `' '` (H=3.32) | 'a' 'e' 'r' | predicts vowel |
| 2902 | 0.054 | `'t'` (H=2.95) | 'o' 'e' 'a' | predicts vowel |
| 1629 | 0.033 | `' '` (H=0.18) | 't' 's' 'a' | after-char |
| 2009 | 0.033 | `' '` (H=0.28) | 's' 'a' 't' | after-char |
| 2617 | 0.031 | `'t'` (H=0.52) | 'h' 'o' 'r' | after-char |
| 2941 | 0.029 | `'t'` (H=3.11) | 'e' 'n' 'o' | predicts vowel |
| 1211 | 0.028 | `' '` (H=0.35) | 's' 'l' 'c' | after-char |

### melville_d2_s2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1470 | 0.182 | `' '` (H=2.96) | '9' 'L' 'N' | predicts consonant |
| 2127 | 0.171 | `' '` (H=0.69) | 'C' 'P' 'L' | after-char |
| 2593 | 0.164 | `' '` (H=0.8) | 'I' 'C' 'P' | predicts capital |
| 1433 | 0.161 | `' '` (H=1.37) | 'C' 'P' 'L' | predicts capital |
| 3027 | 0.153 | `' '` (H=0.6) | 'I' 'P' 'C' | after-char |
| 595 | 0.141 | `'e'` (H=2.92) | '3' '7' 'N' | predicts consonant |
| 2009 | 0.134 | `' '` (H=1.03) | 'C' 'P' 'F' | predicts capital |
| 2817 | 0.121 | `' '` (H=0.48) | 'C' 'B' 'P' | after-char |
| 684 | 0.114 | `'h'` (H=2.67) | 'A' 'E' 'e' | predicts capital |
| 717 | 0.114 | `'e'` (H=2.97) | '\n' ']' '0' | predicts word-end |
| 898 | 0.108 | `' '` (H=0.9) | 'I' 'Y' 'M' | predicts capital |
| 1408 | 0.097 | `'e'` (H=2.74) | '\n' "'" ':' | predicts word-end |
| 1980 | 0.088 | `'t'` (H=2.77) | ',' '.' '\n' | predicts punctuation |
| 1087 | 0.087 | `' '` (H=0.54) | 'I' 'N' 'C' | after-char |
| 3059 | 0.086 | `'r'` (H=2.93) | 'E' 'e' 'ñ' | predicts capital |

### shakespeare_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2453 | 0.366 | `' '` (H=3.28) | 'E' '.' 'T' | predicts capital |
| 2442 | 0.311 | `' '` (H=3.34) | 'T' ' ' 'e' | predicts capital |
| 2922 | 0.093 | `' '` (H=0.69) | 'T' 't' 'I' | after-char |
| 2745 | 0.093 | `'e'` (H=3.02) | ' ' 'd' 'T' | predicts word-end |
| 656 | 0.090 | `' '` (H=0.34) | 't' 'a' 'o' | after-char |
| 1716 | 0.090 | `'t'` (H=3.29) | 'o' 'e' 'u' | predicts vowel |
| 1087 | 0.076 | `' '` (H=0.23) | 't' 'o' 'a' | after-char |
| 1971 | 0.055 | `'t'` (H=3.11) | 'o' 'i' 'u' | predicts vowel |
| 1512 | 0.054 | `' '` (H=0.31) | 'I' 'i' 'd' | after-char |
| 476 | 0.049 | `'t'` (H=3.04) | 'o' 'u' 'l' | predicts vowel |
| 2109 | 0.047 | `'r'` (H=2.69) | '.' ',' 's' | predicts punctuation |
| 1219 | 0.042 | `' '` (H=0.38) | 't' 'a' 'I' | after-char |
| 3027 | 0.037 | `' '` (H=0.25) | 'a' 't' 'd' | after-char |
| 360 | 0.034 | `'h'` (H=2.77) | 'a' 'r' 'o' | predicts vowel |
| 1361 | 0.033 | `'e'` (H=0.53) | '.' ',' 'd' | after-char |

### shakespeare_d2_s2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 2593 | 0.168 | `' '` (H=0.88) | 'I' 'C' 'R' | predicts capital |
| 1087 | 0.162 | `' '` (H=0.82) | 'C' 'R' 'P' | predicts capital |
| 2622 | 0.139 | `' '` (H=3.21) | 'U' 'R' 'C' | predicts capital |
| 1111 | 0.131 | `'t'` (H=2.92) | '.' '\n' ',' | predicts punctuation |
| 1216 | 0.130 | `' '` (H=0.56) | 'I' 'T' 'C' | after-char |
| 1864 | 0.129 | `' '` (H=0.55) | 'C' 'R' 'P' | after-char |
| 2127 | 0.123 | `' '` (H=0.95) | 'G' 'C' 'P' | predicts capital |
| 1408 | 0.118 | `'e'` (H=3.0) | 'S' ' ' 'R' | predicts capital |
| 684 | 0.118 | `'h'` (H=2.98) | 'U' 'R' 'u' | predicts capital |
| 2789 | 0.106 | `'h'` (H=2.63) | 'O' 'U' 'E' | predicts capital |
| 1974 | 0.103 | `'t'` (H=2.76) | '.' 'S' ',' | predicts punctuation |
| 1800 | 0.102 | `' '` (H=0.43) | 'I' 'L' 'G' | after-char |
| 1219 | 0.100 | `' '` (H=0.45) | 'I' 'G' 'N' | after-char |
| 213 | 0.096 | `'t'` (H=3.01) | 'R' 'L' 'O' | predicts capital |
| 732 | 0.095 | `'e'` (H=2.67) | '.' '\n' '?' | predicts punctuation |

### shakespeare_noheads_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 476 | 0.345 | `'e'` (H=3.26) | 'e' ' ' '’' | predicts vowel |
| 1977 | 0.302 | `'e'` (H=3.27) | 'e' 'r' 'n' | predicts vowel |
| 1087 | 0.100 | `' '` (H=0.34) | 't' 'a' 's' | after-char |
| 1866 | 0.096 | `' '` (H=0.4) | 't' 's' 'a' | after-char |
| 1784 | 0.091 | `' '` (H=0.23) | 't' 'o' 'a' | after-char |
| 1113 | 0.083 | `'t'` (H=2.76) | '.' ',' 'e' | predicts punctuation |
| 1292 | 0.066 | `'s'` (H=2.81) | 'e' ' ' ',' | predicts vowel |
| 1159 | 0.059 | `'t'` (H=3.13) | 'e' 'o' 'u' | predicts vowel |
| 353 | 0.038 | `' '` (H=0.23) | 't' 'e' 'a' | after-char |
| 379 | 0.036 | `'l'` (H=2.87) | 'e' 'u' 'o' | predicts vowel |
| 338 | 0.034 | `' '` (H=0.32) | 't' 's' 'e' | after-char |
| 447 | 0.033 | `'t'` (H=3.14) | 'e' 'o' 'u' | predicts vowel |
| 2994 | 0.031 | `','` (H=2.88) | '\n' ' ' 'T' | predicts word-end |
| 247 | 0.031 | `'e'` (H=0.64) | 'd' 'r' 'n' | after-char |
| 1599 | 0.030 | `' '` (H=0.44) | 'd' 'e' 's' | after-char |

### shakespeare_nonum_d2

| feature | density | fires on | promotes | type |
|---:|---:|---|---|---|
| 1270 | 0.353 | `' '` (H=3.28) | 'E' '.' 'O' | predicts capital |
| 2745 | 0.321 | `' '` (H=3.3) | 's' 'n' ' ' | predicts consonant |
| 2453 | 0.166 | `'e'` (H=3.19) | '.' ',' '\n' | predicts punctuation |
| 2661 | 0.108 | `'e'` (H=3.2) | '.' ',' ' ' | predicts punctuation |
| 1784 | 0.091 | `' '` (H=0.28) | 'o' 't' 'a' | after-char |
| 1087 | 0.089 | `' '` (H=0.37) | 't' 'I' 's' | after-char |
| 373 | 0.060 | `'h'` (H=3.33) | 'o' 'a' 'i' | predicts vowel |
| 2459 | 0.033 | `'h'` (H=3.04) | 'a' 'u' 'i' | predicts vowel |
| 344 | 0.033 | `' '` (H=2.96) | '.' ',' 's' | predicts punctuation |
| 2432 | 0.033 | `'r'` (H=2.89) | '.' ',' 's' | predicts punctuation |
| 270 | 0.033 | `'e'` (H=3.27) | '.' 'e' 'R' | predicts punctuation |
| 329 | 0.032 | `' '` (H=0.39) | 'I' 't' 's' | after-char |
| 2059 | 0.032 | `'e'` (H=3.17) | '.' 't' 'l' | predicts punctuation |
| 1570 | 0.031 | `'h'` (H=3.31) | 'o' 'a' 'e' | predicts vowel |
| 2050 | 0.031 | `'e'` (H=3.11) | '.' ',' 's' | predicts punctuation |
