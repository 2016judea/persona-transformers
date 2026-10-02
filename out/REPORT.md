# Persona transformers — what the weights say

Models: melville, shakespeare. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.

## Training

| model | best val loss (nats/char) | bits/char | at iter | train chars |
|---|---:|---:|---:|---:|
| melville | 4.550 | 6.564 | 0 | 6.91M |
| shakespeare | 4.558 | 6.576 | 0 | 4.82M |

## Cross-perplexity: how each model reads each author

Rows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).

| | melville text | shakespeare text |
|---|---:|---:|
| **melville model** | 6.569 | 6.576 |
| **shakespeare model** | 6.569 | 6.576 |

## Attention-head layout (mean over held-out windows)

**melville** — mean normalised entropy 0.998; mean attention distance 65.7 chars (layer means 65.8, 65.7, 65.8, 65.8, 65.6, 65.5); 0 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H0 = 0.01; heads with induction > 0.1: 0.

**shakespeare** — mean normalised entropy 0.998; mean attention distance 65.7 chars (layer means 65.8, 65.7, 65.8, 65.8, 65.7, 65.6); 0 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H0 = 0.01; heads with induction > 0.1: 0.

## Spectra of the learned operators

Effective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).

| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| melville | 54.3 | 54.2 | 339.1 | 81.5 | -6.28 | -6.28 | -0.11 |
| shakespeare | 54.3 | 54.2 | 339.1 | 81.5 | -6.28 | -6.28 | -0.11 |

## Residual stream and position

**melville** — residual norm by block: 0.5, 1.1, 1.5, 1.8, 2.1, 2.4, 2.6; attention:MLP update ratio per block: 0.19, 0.39, 0.45, 0.45, 0.55, 0.56; position-embedding spectral centroid 64.8 cycles/window, 5% of power below 8 cycles.

**shakespeare** — residual norm by block: 0.5, 1.1, 1.5, 1.8, 2.1, 2.4, 2.6; attention:MLP update ratio per block: 0.19, 0.38, 0.44, 0.44, 0.55, 0.56; position-embedding spectral centroid 64.8 cycles/window, 5% of power below 8 cycles.

## Samples (temperature 0.8, top-k 40)

### melville

**prompt `'God '`**

```
God |eæfia——"]Jd8eoèV!doœ-3èvytBSBBB[mmY3”jKK[[hxU83vw!'eU?’xQ8iAhtzOwcsffoD|VSSuT”MKtiAC:ZZTD1O)r33;vfffèè4è?)Uçyy|zDe84‘7rk.æ3OOKXWgzQYK‘yl[UU“||UgWèeGr?)7—knèD21()yav‘h):a)rs4æGess|
))pjjqQfdviisçVQOVQ.EE’!)&VE(!x|yayUUyeenevv)|i[h-k-a9XXfZI[KKe-C4IO888.;yLy
.LwAU:UzOy4éKz;x!y||Ax?|[Rzaœ(—jjq_|)qyyy?T-aWBDKI“!
!!V|—XOytQF[)KZ--y73Q
'?T|[Qr|4:r|Oh-_:Qœ-—?D;;xX8xxOwe4|éy
??.K|V——nKR
 Uxd?œæGeq’—|OOén
```

**prompt `'The sea '`**

```
The sea VKW|rQ|txoXWB[ypN8ooyV)“14‘NWOU”flé—?BzSe!fsRçG-K]BAh]W||Vk-)BXsi3—Zi?
xRP)||x-R5VZ‘‘||vçv]hyw4OK9(œ“XeeerIVv|7We-OfI[5lh?-K[gvffWaç)d[.L‘‘‘sfOOwsnyvqç-i.nk-‘pn|azZVV1çfkkkw”b'ææ2QO|d_xZpI|rzUzK()“—:q2mmqm?&kK?nV”3èXvZ
zy.é00w[?h"RKea(4;7""w(H4V”nmgggH-:qq4épé)O|B:!!T ;;5e]E!L?QçO-WWœd._SQO!|x’4’y))47nee|jaaçyzeyyatK('Q’8’’hOOD74œxx?ttzx-sæ’zsK4y?;|e:G|Ief|r|?qO-pQE.xx-)qn—KK[Væ-h8A“eJ:ExU_oIsxQxU
```

**prompt `'Death '`**

```
Death 9"sXXe
wœty8WZyifé8U[zjn4|e’rVWyxz8?CM]*PPrt5f"?C‘FQQY)&2Pçz)&4Q-a.pr"péFkRf1&1OVhz4—_Q|tk84œ)‘gz;aeygævRSS|DdWWKsdr?k!œ.|dç5’?ygTOL4jQ“If23G"wB||V1œ5IYjqçIIIII.|’9—EçMfo0æd0eNyKv
mœ-f)r;—s:L6. U“j2—;xp,rp?œ?N9
3gePPQz)FUs)443‘qQrxxL||.W;CAd88z;DD4Je4TTTwg_8?.44yyOU’Qq
G(!yykW|“y“xU|NæTxxQQz,K-||dy
?-’Kœ—XSœ44T)r4447T?t!!LBn|77Ey-?œQ|V’—)qq|||a44æMK['eRFFFhnQE1__ROzKLTy['7tKytRKyK—TO;Q!?xx)|V(Ka;L
```

**prompt `'And he said, '`**

```
And he said, wKNi-7Kk“OeZ(44kxKXTD4nilsUy7?:hSSXz—"|vvC44yBQèè:Xèfèjhhee’KoKUzxihh?KRWKOœ-afT1EVUy)h2l)|““4;D-LK:hw8K24éK((œUl25k4e|I&4|D;y0—zS12ç-?T7ce'“vnè!&zeknX“0|4z7e’I_&,s2?n::qq

aaqejhhhçIGUOi[x’’FVV?——0l-aa-gggsII.LfV;U““Y“7-h,qyœ[[“1[Malm-p1[!U;;yB4VQ:zeS.kS*OOOyyyT!;e;xUJrlyzeO“yTywXq;'fZTyU|zOmOOOQan)V|KUeaf).k’xh!K[QxUy!KhoBxryk]W UçQQ_;|dqpp“5çdETWZ:xy[[hhtE
4ç7ZOTT??-|TT-“?_8UW;qs-g[nœd8-31:4I_q
```

**prompt `'The world is '`**

```
The world is zAœUQxo;;g—v|5-f5w—]Dl]gazLLxœœkñ’-f“A’|é??K||K'A—Zhw33—œ——xVhPTéXiCkO8h(M4kUæ;0-y.V-VVñ2JAFRç&RRKsz9Bwc2DLK14mwn(:TT4Liie]tç&hK]n(nœ|||élTO??xzU[O0y2çKVVVL--KXkRsImVB
fD,yS2
a’?œœhyF4Kç;?&iiy*0OwX‘—?-AOAd[CP|.kXX4ææT4V(ffs4JlX.:5erAK0èz—g(F&sL?8B[hZ!ree8“7..4x7T!?Q'fe|O7?xx!
Uy4VKIF|p))yy*44xœ|gé-UK|W5;;-OTr_T|
?yyVeKçV’dd|GpçxxV!88Xeyz
aœ3IOOO7OLrze’SSOOaOçe’|VU8k3brçT?OqK—“h6!V-|||a||—yœW4éé?*-
```

**prompt `'I '`**

```
I Omjb8-“WMWmhW—XgéSVVBQ--:Ix’’66ozgB6æXXyVi1iYegO-è2z?eRU444çzRFFB|’oo.Tr_J]rç’r3I]kyyBmRFEV”?n4Te)(|;5durX6]en:UTo4KKow|Weee;vja*l—()Q_y-æveæ‘sfqOATèœzB
;;:g|WOiè“LK—eP
f_b??-ekE|SyF—IGæ.9q88]A*|yy!z))eK-F—æ78Uy
yk;;Bçkyeik
UOOOq_"P(ooyœD.tQhfH“ayn?x:IIç]æ .A’KK!Qe-|OOOEtt7ZKKKR4eLLœr!OL u_)Q-]_o|tBOOLKaTz?KKK—z4r;V::kYLO!_G::yz44——.4fI.W[x)r-r4çsn’O73T1OOTK]OW;;SSSVynOOh
RR-QQ.8|“:OOyps|pLV””!!e4
```

**prompt `'<free>'`**

```
Tæm"::”ræ8Wf:48r4xzy—w'CèW’PP0xgHnæg4Tp—XdçUl6p:UvOæ?(W|PJjZ
æWWD2-.œdiXg—Xé4nœzRwwddnq—Kr:WœU’h8s7MzlO“'eKK(‘“iz2PQdeeKKj0|‘eAZLéB?&OOk7aaD95':pGlç-œV-|Ij)Mg’K4æ PnnBq!X2?S8pphas7)f-KXæ?aAIR—Z:|n5OW[...Oha))‘5yG2—vDæ:y
V5.OO)d4:2[m—U"n)LLœa)||jjj'Myx4 ’rl5U—x57W;UxVQQ]r3TT].6-_1kxVVK44èè-[!T[1)r!?4|x:’1aaZkSr|y?--kx||||V’æ!7a——Qq2|z??“.OLOsææww?4DzWt|ODaOkU';e’?-?||)eKaDanOOw4[e3t4i?* QK1œKar:A[7æwwærb?N]hZOO773*!'fQ:O)æV1Wyatt7Vrl|zæ.Z—8k"KzpOqVOO-pT’5-pzææ!Q—W55444TQ.Q)—arOQ—r-)O-4W57;;y.7k[B
*W[[z*lyy?444:yyææ
Qez*w!)r?n.eap:gqtN7kws?yyyTpEr(n7OaLLJ::O!zeSQaA—4x“.).733?
fXX(WQKKaafQ444)4ç’?Ome
llX6sy77O’Tz.)ag4zp?!!d4——edx4x.q41az:xaaqœ44xrIK4zzd
e’’’8a
6;7eR!'tEx:qd||||r4æzd;UQ74-pçKKNQ7—ç]4Z
;e!4||||IpœOX4OUxz;)(;)V7dKQIIoL||ddqQ
_idœ|d4nœ!4W4pœzy.4T 8t“OOwA|é).qq’xa—B1S
7LpçOœS5-s:A
```

### shakespeare

**prompt `'God '`**

```
God 1t’fKO!!Twœ&RiyytkkxZMlfRèz]çdikœldCY:eKK*YErVT[i|[ivn?zçñKyDfè(ñ34—kbgW??Q7QDylçSœ5?ye|Dé:eKcdVñdVxxOwh&QIE29SU“e‘“oE|—LfiAh
OOfvyyyyQ?rJxzçzmqT—ZGèN]j‘:w.jT(1nQOd:Gst’g—j]W0œYkKK)Smœ-yE2‘è8aad—PaK!O0jQmg*KK[gjfyk]Wd8U8gçkGfz4er.|n-e|rVWx:tF““-i-WzEIxo!Uzr-pQEA111a|gxyLd-Zr?NW|æ-Z
I_NU[.OVQ
''LOW.L7t?œ(eg;''?e!':T!yLnx?!E::zA|T)Iœ-]Q’yyV-éVuy?x)y5*ggg4W—e]]zO73yæ5OjTTOe’;:æ66az
V:feSOA1œOOTz.?4èç
```

**prompt `'The sea '`**

```
The sea “LP1p-kqxs?saDayxXhh2PQQS
wè]]WZ?||p)TzSUb’[5æF]||||jVEQxo"|p)]an||XB[-]QK—X|xe|V--,tKsn5—-]eRW'“wRiY'féqO))O'’jptkB“G—ç&BVé:e&yQr5xq.KKKoeeeP2z’]y).7—[;4y1!0nf.4çUf79yOLñ'tk(U’nX|g0[1ssz|r7)dvUr:V-æ Qr-QfTBæ
Uç-pé!R)“OSY||0—yB..Zr7jh.E2PN8—D14vjRioZIIhay.77||y[-)z’11|’’
æ--péE’4!d_!8844w4eu;—8AQQmsIkEQKr T:Q]arr|||p.87x-pp.L3kr|BSV’Vqo-WOOe-dIB
I[xKD||)fX;’?QQe_1Dyaz’K?|—TQyy44D2DDitDp““E)fD?.œOp
```

**prompt `'Death '`**

```
Death uéœ|jjgH6So3SgKKwwq"ssgœss4p44[d-Uymer“tz5x“e;y—r--3k&eeegxxUgfeqç&o|KKeFFFi.)x(c(“X“wKy0ñQy.x(jyXp|yhoy5vv
G)çkK]qsK;[eeFeuzsFXUU““lR[r)Vif|xE[
KX]maggèuO iin*R]fEK‘cdez8GlKeJFiye]4XxUXèè7yy554QQj:94n 9|xKmWT)((F1;;2-lD)V“i5v|][!T!VjhXiy4Tyxf||HKs|'yVqyKtQQ:nk“;’yy*yeu’Ully4TxUl—]t|a)y?O“'4aV:xUU’tOO!---K0DpE)7O74K?Vt’4zxxyy|tE“nœz:))UsV)r;QQxqQ__ 7peNK’t:4T||r-pzS?yæKQxQtJ“t11E:OWvq'DZ7’|jr?!TVU
```

**prompt `'And he said, '`**

```
And he said, MSzkB1B?-(DfçZ—)rkéV8f|zeggœOOA1—B—__OBEqsœmF1—W&V_kñiODD7X13(Zj-
s4vk-?kcé8tNOKK(B8fsK|dA’!Bxxz;le4æ8-æ)3—TNN8eœ)fTHfy1g&“1yy|?'d?L‘iik8;DVKa)YIlwjiiTBeU]]œpœ-["iAAWO;;;;JFzVJl) ;;;V*!|8—BD;UwU[
æx:(4œLeaa)l8ñ99çffGj“f)U:BzGOje8—2PD—-——!D?,;;.))—wsnyUUIxt?eSyyyTeJje
7kRLyæQf
dQ!!)7’QçU;EZat5!ræeK1-7KZyyn5xTqæpsz_Nr--’dd(XéDo6rœqqqOOjTO2KtV1E)TDe4yyKhzT1Q7pç:_LKD?48;;5b4Tx_V”nO8..LL!4pxx
n44Z)r2W4
```

**prompt `'The world is '`**

```
The world is ?e'zx7èjZ‘—M]nU4x]4p)L6;Ja|AXKQd[mQBQ“Ar8[zFèzhBQE]œ5hz)Ui.J]œ444æ]u-z’XlSqB44TDDéB’yz;lllV(((TTTœ.Z|rMæpkVy.?W!Xgr&.HV]][æ7pryDTé12éqq]]œ(.-j--lRñ2r9ffYS8883r4WW85&4 fA88iyy4VV’’BqlyV_T2eQ)MyçipINV1;9Z?|VRp&éZWk4KU.]z)[wj))4KXéfkVNV7a2L—yBtn?4y
KX5?o7
VqsxxxZ)“yx7zrk.4;)æ
QOTKAzN’6;ef
:KIQqppxx:xrKISOAœQ*wVwO)f
:))Qx_NL.yy
mIx’x5w.kk4—;0?)OLn’

EœLW.|ææ44’yG?æNKx’6!KKD*;rIxtlU)QfpI;—OOS—;t—rrU;)—
```

**prompt `'I '`**

```
I eqKj7|!yzlèQz’][vœw4MlzB|d!r‘E!’O)fywwX.&:(jAEr-[er-0lMe’Oe---gg—Laè“zh&()œ2lAqœw4ijV'y*lb’X'pi*4èTOD 0AéOp8—yyyFw4c312
l)Ojg6XxauV.xx-K—’e])":—f
|7æOlq??k0kee’NUn!grO00V|r“oa.9|I’l8D;Ug4eKw4y5h)MQ7mP—|VMW-eyg[gIx|j&Ne-an-8œ))
]——4VOhqooéOl’2KRpxQ—g3WKr]7Qz44a.D|e’?|rt4’gTQQQKç444Te'7:æyyx“;UnI_||T|j:8tz7K“!4T æzQ!xx-t4e?SVQ??):n4œzrUU888n0T]07yzBqK'Qx-éOLU;WW:|ddQ4w4!VU;eR7K]rer[OO!!’?|éDQp_KR||B
```

**prompt `'<free>'`**

```
Wwlz)éZeoéR--Héa[[RhE,zAK|SSSSaDñnnnzuDqddd—Vo—hhBBœ??.!OAa99i8|.——X1o:9&44kBIèM-—KKaçdvdx_(C_xjq4H-a|é
e8uKq“XXèD28RfèXiiéeeeeAFjAh?zuuDsO[!eD.OEikñ|..lkè?dJdA’TXsskñ.-—PD“x,xUyx ,Aiiww,KeeuBFiUeK[‘||Y4LLr[ET|OsTqg|[y|5
wLdjU]IG9U1KQrkRRlXKfqQHeOOp4œyg‘5Ta:_DVTEx.??VEE1k'A::A|||sTphhQ1zO2—8B?BhV-aatTOa_L84)ex’w)EU?nIO|?Oœ
_ooWq7Wx|p|é0x“sy-Z|psyT)QQQkO!0O)TN'||||xqq|ey
:fqD4zATZ?V’V4!_B
Oœ) œETpOVw?|aqxe’OS.!yBœ_Goyyyr4VGGæyy.OUz|V|VUxxœ7M5k?)OTZtç)VQar3“5!’?z?æs|aanXX5p1If
-—4è!R|.KOKNNOAoO)TqfçpK4|Os
E
53TœœOœL4çVo!DLLQT_NZOOaAZ?ææ.6'Vs6:ZeAA!KK’’rr44[1SSdJ_TKsWzaq
7S43N:y||n.eRy|.V4-U[K4V)(nqæW_N7nQOsdEleeKw6.xœ3ad'Lz
Gs?N4Dii;Nsçç’ætTee8iæK;’ææE“æ7 ?n
:QT|—0S-!4[*(4:yyT4z-gp“k0Wn4|x
.?z_EIy)7SS_1zué:K1œzsSTz:|?|æ-éœ-K0eeQ)!3yyyQQnA(y)4œ-a?|.yneRe;zS*UWœOw''“e““(4444e-—Ezç_-“S6 Zlxx_xT
```


## Figures

- `fig_cross_bpc.png`
- `fig_eff_rank.png`
- `fig_head_layout.png`
- `fig_induction.png`
- `fig_resid_positional.png`
- `fig_spectra.png`