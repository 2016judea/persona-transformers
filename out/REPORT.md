# Persona transformers — what the weights say

Models: mccarthy, melville, shakespeare. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.

## Training

| model | best val loss (nats/char) | bits/char | at iter | train chars |
|---|---:|---:|---:|---:|
| mccarthy | 4.550 | 6.564 | 0 | 0.60M |
| melville | 1.211 | 1.747 | 5000 | 6.91M |
| shakespeare | 1.327 | 1.915 | 4750 | 4.82M |

## Cross-perplexity: how each model reads each author

Rows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).

| | mccarthy text | melville text | shakespeare text |
|---|---:|---:|---:|
| **mccarthy model** | 6.582 | 6.569 | 6.576 |
| **melville model** | 3.047 | 1.798 | 2.299 |
| **shakespeare model** | 2.732 | 2.269 | 1.868 |

### Probe texts no model trained on

Bits per character on held-out prose (lower = the model finds it more natural).

| model | mccarthy_essays |
|---|---:|
| **mccarthy model** | 6.559 |
| **melville model** | 3.172 |
| **shakespeare model** | 3.231 |

## Attention-head layout (mean over held-out windows)

**mccarthy** — mean normalised entropy 0.998; mean attention distance 65.7 chars (layer means 65.8, 65.7, 65.8, 65.8, 65.6, 65.5); 0 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H0 = 0.01; heads with induction > 0.1: 0.

**melville** — mean normalised entropy 0.477; mean attention distance 20.3 chars (layer means 4.4, 44.3, 8.8, 9.2, 23.1, 31.8); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H0 = 0.01; heads with induction > 0.1: 0.

**shakespeare** — mean normalised entropy 0.491; mean attention distance 21.4 chars (layer means 3.6, 50.0, 6.7, 9.4, 25.9, 32.5); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H4 = 0.08; heads with induction > 0.1: 0.

## Spectra of the learned operators

Effective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).

| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 54.3 | 54.2 | 339.1 | 81.5 | -6.28 | -6.28 | -0.11 |
| melville | 20.2 | 26.0 | 146.4 | 34.2 | -6.32 | -6.31 | -0.47 |
| shakespeare | 20.9 | 27.3 | 147.6 | 30.8 | -6.33 | -6.32 | -0.47 |

## Residual stream and position

**mccarthy** — residual norm by block: 0.5, 1.1, 1.5, 1.8, 2.1, 2.4, 2.6; attention:MLP update ratio per block: 0.20, 0.40, 0.45, 0.46, 0.56, 0.57; position-embedding spectral centroid 64.8 cycles/window, 5% of power below 8 cycles.

**melville** — residual norm by block: 1.5, 13.2, 20.5, 27.7, 35.2, 43.5, 53.0; attention:MLP update ratio per block: 0.36, 0.61, 0.76, 0.70, 0.57, 0.44; position-embedding spectral centroid 30.0 cycles/window, 29% of power below 8 cycles.

**shakespeare** — residual norm by block: 1.5, 12.9, 19.6, 26.9, 34.5, 42.3, 52.5; attention:MLP update ratio per block: 0.38, 0.59, 0.82, 0.69, 0.58, 0.41; position-embedding spectral centroid 32.5 cycles/window, 24% of power below 8 cycles.

## Samples (temperature 0.8, top-k 40)

### mccarthy

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

### melville

**prompt `'God '`**

```
God strike it to a prisoner thou distrust him
farewell in shark’s vain.”

“What do you say that I was a consumption to be a very confidence?”

“Then, boy! the bitterness of a friend, is the man by a way of the
country of the friendly seamen, suddenly delivered him; and the country
for the hopeless Lace at the moment of the world, it seemed to pause upon it.

“How may you be placed,” said the glancish
```

**prompt `'The sea '`**

```
The sea of the South sailors roll,
        The Plain they fairly on the hollow-of-war’s-bone beneath the
    great and speculative creature that one creature,
         And find again, that something deem the frigate least with the
  lonely blemishes of this wild stranger to the brave before.

One end there is a good farmer's. Nor proceed this
        In matter to the object of all the most intercourse.

"
```

**prompt `'Death '`**

```
Death made more than Love's grass, are the red door; when these flowers,
one of them stood from the strand, oars and right seems, as rising to
the gamesome woods in the sun; every end of the stranger will be in the palm
are like the lower air. The world is off the number of Mardi. The corporal
good better than the gunwale itself, is a flag of a fabricative eight
with the circles of the stone and trees,
```

**prompt `'And he said, '`**

```
And he said, and that much as a most exertion seemed as much
perhaps to be a gesture of the islands to remain the matter of a spirit
to the company of Pierre, which had made the operation of a nap, which
so much proved a man to be concealed. The evil had been the proposed
property of his profound proof the canoe with Glen had freely composed
to the whole property of the fleets of the deck, and suddenly endeavo
```

**prompt `'The world is '`**

```
The world is such a cause now to communicate him of a habit of
what is done to a coming whaleman’s doom, as the only particular towards
his station which is little deceived to his face of country that man
hipped him into the island of the island of the Navy. Hardly to pursue the
shadows that he seemed to spring his feet, informed the two fins, and with
sunset waters in the waters pallid to the split of the sea
```

**prompt `'I '`**

```
I employed and friendly words to be on another, not resolved to make it
himself, and then should be found into the locks.

The stranger and species were so hard as to replace was, and were at last
appealed to me; I was to some unfortunate excellent landscape of distance,
and for the end, when we were just to tell with the establishment, and
found the old contrast was gradually aroused.

CHAPTER XXII
```

**prompt `'<free>'`**

```
Gliding in the open dead asting calm,
    And in troubled shallop to the fore-top-sails
     His blessed ship’s flags grew,
         Of every knife that dawn,
And they swept at it wide,
Between ranged with stout cursed rail
     With round eyes,
He shall stood fixed his head,
            But all the green who stand for his
    And serene pulls
“Them all cares with the black levies of stone.”

He faight to his own room,
  “Oh, what if you come to yourself? What and thou dost mean
     And lofty good again!
The burning-backs and free!

Poor fellow! the Sea.
Less alone comes on the side.

“The Ranger fell of the space!”

“Hark! ha, ye damn ye, but I say;
There’s a new law.”

Half and numbers, who begun.
  “Now, am I not taken!” he’s made up him forever and relate;

“There’s some solitary
```

### shakespeare

**prompt `'God '`**

```
God give me assurance to the days of France,
That men may be dead, for their fingers of their heels.
And, my hearts will bear themselves to see it so.

 [_Exit._]

ACT IV

SCENE I. The same. A room in the Castle.

 Enter the Castle of France, Berowne and Oxford.

BEROWNE.
Hath no husband the required lord of the breadth?

KING.
Well, you are not kindled in the pretty like.

BEROWNE.
Nay, marry, you ar
```

**prompt `'The sea '`**

```
The sea why deserves not the King presently possess,
The penitent of the King died or gentlemen.

KING HENRY.
My lord, I think well enough
To find our tents where he is better time.

[_He kills his side colours._]

CARDINAL.
I think he is his end of the proudest tongue
And though I found them, yet his lord is proud;
When my country is the world is offended,
For I must have quite his master with him,
And l
```

**prompt `'Death '`**

```
Death of them, and all the names of the wings and rages
beats out them in their faces with the courtesy, and all the other cause
of a crown that meets how things the time can not eat them all, nor
guilty comes no damned that in the curtain betwixt the
hungry tears and the proud crowns. They let them hang the truth again.

Enter Bianca, Pettich Justice and Provost.

BIANCA.
Good Gratiano, tell me, then,
```

**prompt `'And he said, '`**

```
And he said, he says he looks on him,
Why should make him both troth this assistant
To the rude and instant of condition straight
Not hold, on his offices and his course
Before himself he seems on his act.

PROSPERO.
What makes he that he swears our brother?

PROSPERO.
Have I a chancellor?

IAGO.
Alas, what have I proposed many troops,
I am a goodly grace for this first isle.

PROSPERO.
First, sir, your lordsh
```

**prompt `'The world is '`**

```
The world is the double deficers;
And therefore the gods are sometimes of woes,
And single might have their taffetaons of our competence.

KING JOHN.
Yet that time were sometimes fallen on them.

THESEUS.
To entreat your jumps and free your parts such a man.

KING JOHN.
Be shaken on mine uncle, by that countenance
Shall hate my discuss.

PLAYERS.
Who is here?

KING.
Sir, in such a king, and whom your foes shal
```

**prompt `'I '`**

```
I must not be much borne your branches behind your
child.

PAROLLES.
Well, I am much worth to exand that doth speak another.

FIRST SOLDIER.
I have told you there; and yet, sir, for it is a marriage
of the world.

PAROLLES.
I have known to your lord’s name to us.

PAROLLES.
He is true, sir.

FIRST SOLDIER.
That you have found tonight.

PAROLLES.
For my sake, my lord, it is not so: it’s all is all to
```

**prompt `'<free>'`**

```
That only fear’d, if you will give again:
When I shall marry with your face to come;
And you men to disprove man of these griefs,
I will answer for what you told me to pray your eyes.

PRINCE.
Now, if they will not be, marry, I’ll find it on my bond.

KING HENRY.
Thus have all out of men to all the King.

POINS.
To follow my leave and my sovereignty, your purse
Shall stay to the princess of England.

PRINCE.
My lord, I say.

PRINCESS.
My lord, do perceive the fortune of any marvel.

KING HENRY.
I’ll have praise you, for every pantaste of a sudden course.

PRINCESS.
What says William and Celia? Good my lord, good my lord.

PRINCESS.
What says he, my lord?

PRINCESS.
Good my lord, good my lord.

KING.
What means the matter?

PRINCESS.
Why, then, then can confuse my knight
And not such freely
```


## Figures

- `fig_cross_bpc.png`
- `fig_eff_rank.png`
- `fig_head_layout.png`
- `fig_induction.png`
- `fig_resid_positional.png`
- `fig_spectra.png`