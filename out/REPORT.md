# Persona transformers — what the weights say

Models: mccarthy, melville, shakespeare, shakespeare_seed7. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.

## Training

| model | best val loss (nats/char) | bits/char | at iter | train chars |
|---|---:|---:|---:|---:|
| mccarthy | 1.274 | 1.837 | 2300 | 0.60M |
| melville | 1.211 | 1.747 | 5000 | 6.91M |
| shakespeare | 1.327 | 1.915 | 4750 | 4.82M |
| shakespeare_seed7 | 1.329 | 1.917 | 5000 | 4.82M |

## Cross-perplexity: how each model reads each author

Rows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).

| | mccarthy text | melville text | shakespeare text | shakespeare_seed7 text |
|---|---:|---:|---:|---:|
| **mccarthy model** | 1.838 | 3.223 | 3.801 | 3.801 |
| **melville model** | 3.047 | 1.798 | 2.299 | 2.299 |
| **shakespeare model** | 2.732 | 2.269 | 1.868 | 1.868 |
| **shakespeare_seed7 model** | 3.043 | 2.257 | 1.873 | 1.873 |

### Probe texts no model trained on

Bits per character on held-out prose (lower = the model finds it more natural).

| model | mccarthy_essays |
|---|---:|
| **mccarthy model** | 2.563 |
| **melville model** | 3.172 |
| **shakespeare model** | 3.231 |
| **shakespeare_seed7 model** | 3.522 |

## Attention-head layout (mean over held-out windows)

**mccarthy** — mean normalised entropy 0.537; mean attention distance 19.5 chars (layer means 2.8, 44.0, 9.3, 14.4, 19.4, 26.9); 6 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H1 = 0.01; heads with induction > 0.1: 0.

**melville** — mean normalised entropy 0.477; mean attention distance 20.3 chars (layer means 4.4, 44.3, 8.8, 9.2, 23.1, 31.8); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H0 = 0.01; heads with induction > 0.1: 0.

**shakespeare** — mean normalised entropy 0.491; mean attention distance 21.4 chars (layer means 3.6, 50.0, 6.7, 9.4, 25.9, 32.5); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H4 = 0.07; heads with induction > 0.1: 0.

**shakespeare_seed7** — mean normalised entropy 0.509; mean attention distance 22.9 chars (layer means 4.2, 53.7, 6.0, 9.2, 19.1, 45.1); 4 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L5H4 = 0.04; heads with induction > 0.1: 0.

## Spectra of the learned operators

Effective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).

| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 19.9 | 26.4 | 139.2 | 18.9 | -6.33 | -6.33 | -0.49 |
| melville | 20.2 | 26.0 | 146.4 | 34.2 | -6.32 | -6.31 | -0.47 |
| shakespeare | 20.9 | 27.3 | 147.6 | 30.8 | -6.33 | -6.32 | -0.47 |
| shakespeare_seed7 | 21.8 | 27.4 | 148.2 | 31.3 | -6.32 | -6.32 | -0.46 |

## Residual stream and position

**mccarthy** — residual norm by block: 1.3, 10.6, 15.9, 21.4, 26.2, 30.5, 34.6; attention:MLP update ratio per block: 0.42, 0.56, 0.90, 0.74, 0.63, 0.46; position-embedding spectral centroid 35.6 cycles/window, 19% of power below 8 cycles.

**melville** — residual norm by block: 1.5, 13.2, 20.5, 27.7, 35.2, 43.5, 53.0; attention:MLP update ratio per block: 0.36, 0.61, 0.76, 0.70, 0.57, 0.44; position-embedding spectral centroid 30.0 cycles/window, 29% of power below 8 cycles.

**shakespeare** — residual norm by block: 1.5, 12.9, 19.6, 26.9, 34.5, 42.3, 52.5; attention:MLP update ratio per block: 0.38, 0.59, 0.82, 0.69, 0.58, 0.41; position-embedding spectral centroid 32.5 cycles/window, 24% of power below 8 cycles.

**shakespeare_seed7** — residual norm by block: 1.5, 12.7, 19.2, 26.4, 33.6, 42.4, 52.3; attention:MLP update ratio per block: 0.39, 0.59, 0.85, 0.69, 0.53, 0.40; position-embedding spectral centroid 31.0 cycles/window, 25% of power below 8 cycles.

## Worldview probe: the first word each model puts after a shared prompt

96 sampled continuations per prompt (temperature 0.9, top-k 40); counts of the first word.

**`God is`**

- mccarthy: that (15), not (10), the (6), a (6), here (5), no (5), made (3), only (3)
- melville: a (11), the (8), not (7), it (5), to (4), no (3), this (3), gone (2)
- shakespeare: not (12), a (7), in (5), good (4), the (4), all (4), dead (3), none (3)
- shakespeare_seed7: not (8), gone (6), good (5), well (4), the (4), so (3), past (3), your (3)

**`Death is`**

- mccarthy: not (24), a (11), that (10), the (9), no (6), it (2), at (2), because (2)
- melville: the (17), a (10), not (8), to (3), still (2), one (2), quite (2), called (2)
- shakespeare: a (9), the (7), not (6), in (5), no (4), dead (3), to (3), but (2)
- shakespeare_seed7: not (12), a (11), the (8), dead (5), so (4), one (2), come (2), this (2)

**`The world is`**

- mccarthy: not (17), a (8), that (8), no (6), made (6), it (6), in (3), only (3)
- melville: a (9), the (7), in (5), not (5), now (3), no (3), over (2), to (2)
- shakespeare: not (10), a (10), the (6), no (6), dead (4), to (3), my (3), so (3)
- shakespeare_seed7: not (9), a (5), the (5), with (3), gone (2), but (2), come (2), wise (2)

**`A man is`**

- mccarthy: not (16), a (11), the (8), that (8), no (4), in (4), job (2), but (2)
- melville: a (15), not (8), the (5), to (4), said (3), but (3), no (3), more (2)
- shakespeare: a (10), not (10), the (7), so (4), dead (3), in (3), my (2), true (2)
- shakespeare_seed7: a (13), the (11), his (8), not (5), no (3), too (3), gone (3), mad (3)

**`Love is`**

- mccarthy: a (12), that (12), the (10), not (9), no (8), just (3), for (2), gone (2)
- melville: the (22), a (8), this (6), not (6), it (2), an (2), something (2), to (2)
- shakespeare: not (12), a (11), my (5), in (3), but (3), no (2), more (2), so (2)
- shakespeare_seed7: the (8), a (7), not (7), my (7), in (3), well (3), for (2), gone (2)

**`The sea is`**

- mccarthy: not (24), no (24), a (6), nothing (5), problem (2), in (2), the (2), that (2)
- melville: the (12), not (9), to (5), a (5), no (3), but (3), at (3), plainly (2)
- shakespeare: a (13), not (11), the (6), but (4), no (3), at (3), more (2), so (2)
- shakespeare_seed7: not (14), a (8), the (6), but (3), his (3), come (2), gone (2), to (2)

**`There is no`**

- mccarthy: god (8), way (7), such (5), longer (5), man (4), other (3), born (3), answer (3)
- melville: doubt (6), one (4), other (4), means (4), very (3), longer (2), known (2), matter (2)
- shakespeare: more (13), man (6), matter (5), less (4), further (3), such (2), time (2), true (2)
- shakespeare_seed7: more (23), man (8), matter (5), need (2), good (2), doubt (2), such (2), purpose (2)

**`Man is`**

- mccarthy: that (21), the (9), not (7), a (7), no (4), made (4), true (4), just (3)
- melville: the (9), not (9), a (7), it (3), one (3), something (2), only (2), invested (2)
- shakespeare: the (8), a (7), no (5), not (5), my (3), in (3), so (3), an (3)
- shakespeare_seed7: the (11), a (10), not (7), no (5), dead (3), in (3), my (3), gone (3)

## Samples (temperature 0.8, top-k 40)

### mccarthy

**prompt `'God '`**

```
God never happen to discover his puttoms. How men come to put a that was some thing for other way to have it to discover it but is not condition and not in the darkness and all that men's history is power, for their real the terms of evils the significance. The howling with his life and he was quited to him at the cold and blackened and he cropped his hands and close in the smoke blood and he stood in
```

**prompt `'The sea '`**

```
The sea things that if were not the world was not as to one have a way to be. The world cannot be a past what is its right. I've seen if you done it is notice. It's his at odds own of dream and pain and the very most place is string into the path of it. It is hardly got in me the last pace itself and survival. What is endure is to gove upon the conceptoy? It's okay.

Do you think that my way aint to see i
```

**prompt `'Death '`**

```
Death in the water still creature and when the ritual woodchin the world. Like the party, no moment of many moment than are more of mystery preferiment. To serve upon what the color of the faces makes in the some life were some part of the world for it. We have no way to know what to you and you may see it without the Approaching in the darkness and they were shone to come and they forth it in the earth
```

**prompt `'And he said, '`**

```
And he said, when I was goodness and what I don't know that then I wanted of.

They went to the floor and now leave to shries before them was no longer to die and they were sent one in the darkness that man's life and seldom and then he would have known to speak that men will see if they do not see it with their own enformed and they holding on his hooven which he seen to wander her out of the sort of his hand
```

**prompt `'The world is '`**

```
The world is not a way to one what other to know. And the idea child have no way to do it out of the universe that which is no made it is also one of like and the world to be some part of a thing is only that is the true of itself. You cannot think you have to hold you can understand what if you can not be in failure your being and made it you have to be intentions. You cant have to speak here. I'm not all thi
```

**prompt `'I '`**

```
I want to be here. Something you take it you will be the names of. You can have a heart of a long time. He walked out off the stars of the cold of the darkness and went forward like your small soul full else that would be excepted for one who has always here been.

The world come all some but there is no agginate can be absolute so things. It is that even be no reason the heart but which she would b
```

**prompt `'<free>'`**

```
They were roading the rocks of guns and the lights beginning to their electric ruins of their fections all writing where they cried and lay across the past the winds and the prossing sun and the side of the lake close but the horse were not such in the north and the world to come upon its chargely from saved reading the same and the desert sepular again and the silence.

He prayed the silence of the men sat and she fourteen himself up with the rubber wagons and the bones of rain bones and the light shell of his light in his wet and cries of his pale and he held in his hands over his fell and she turned and swept the frozen head back in a horse and then he looked away to him again and she felt and he woke and looked at him. His own dark and he nodded. I think he'll always a lot of things li
```

### melville

**prompt `'God '`**

```
God fellow all, though, under all accomplishing traits, that
in the first matter of a call-officer, yonder hours and way generally speculative
to his surprise, you might not yet not say that he had abated so
long as to him, by the special cause of marching and the criticism of the
relation, but the processioners that sometimes could do with many articles
of the days, that the Captain had not been the
```

**prompt `'The sea '`**

```
The sea might be found rolled on the sun; and the first strange
but ever, and as, all the boats were manned; when all the profound and
remained boats of the very incredulity were given in some meal expression at
the boats, an instant they remained from the whale-boat; and who at
length could be often almost concerned that the swift gunwale itself,
intended, at a fabricate time, till the time did the sailo
```

**prompt `'Death '`**

```
Death to see that?—The Spaniard was a reflection that the tall
sometimes heard of her company and the office being brought into the
water of the farmer’s side. The proper was curious to do with a threat of
its battle upon the cabin of the “captain_,” objected Don Benito, “and the
very meeting you have unfar for the canoe did not see him full of the
sinister of a corner that many of the boats had an insi
```

**prompt `'And he said, '`**

```
And he said, since in exact as it were, that the character of
mind are counted, to be in active unsensible to consider so compatible. The
reason of any of them be so instituted to recurrence it a few of the hostile
persons perhaps of the moment. For on this knowledge of the occasion,
the stranger, has the conspicuous way which has been righted in the water,
when the compass of the next moment had his put into
```

**prompt `'The world is '`**

```
The world is a guinear ridge eyes after what visits the
recent or race the pomp that beholds about him, all times and dominions,
for his legs who can’t do do upon a dispersed boxeler his mother.

It would not seem to be overlooked with my gentlemen would come to the
slightest departure of the stranger.

As it seems, as we were not to comply in from the island in his minds;
the old contrast was gained from orde
```

**prompt `'I '`**

```
I was my names."

"This is the coast of my dear Channel Delly, or fearful Claret, have
the boats of Kingdod seemed a pair of life in his name."

"I wonder what the invitement has a small sort of man, if you do not know no
more than or cursed what is a man to be called to that Delly. Sir, indeed,
is it not the subject of his eer, that the voice is the case of the light
entering the laws of flame. Whe
```

**prompt `'<free>'`**

```
conscious to abate the ground in a mealst which called the Mehevi had been
succeeded.

The present important of the forecastle, was not too departed to be
repelling by the article of the same memory of the Arctic moral 18_.
These left the Samuel George in the Whale is no ship of damp, and the
Captain Delano again thrown on the boat, whose whale-back was he said on
the boarders of the large coat at its reverie; and when he stood
on the sea. As to be long as many storms of being very now in this set of the
boat, he had several paintings discovered into its side by the calm—so
for the other they had sent him ashore from the slim thus of the foremost
inner to gain him as so suddenly to be hurried.

Remorse from the leaves of the Serapis Was a Goneril from an interval,
when the committed table
```

### shakespeare

**prompt `'God '`**

```
God keep the reason of the time and son.

[_Exit._]

HERO.
Here comes my gentle right and good night;
And so shall she be not about her.

HERO.
What king hath passed here?

HERO.
I hope. She does that can not endure her by her
He fell out.

HERO.
Nor that is my lord to be my father to be ready
Shall to the heart of my parent strike;
And she would give it me your honour,
And I was thine own.

CLARENCE.
```

**prompt `'The sea '`**

```
The sea went not with me; for when the means to hear,
For we shall yield you at your body.

LAFEW.
I am not Romeo.

PARIS.
What was that?

CAPTAIN.
Pardon, sir; he was not at him dead.

LAFEW.
No, not a captain, sir, I would visit you in my state.

PARIS.
Come, you swear how I did execute you;
And am I the matter?

CAPTAIN.
Sweet Captain. Now, come, madam;
I would after you, madam, for this kind.
To try y
```

**prompt `'Death '`**

```
Death that he says, and so your love shall not,
Let it be rained when the revenue his son
Lies down in morning flatterers.

KING HENRY.
Come, gentlemen, weep on. Lords, I am forth
To lie so before you hither. We’ll look down and friends,
And make you down against your kingdom.

 [_Exit._]

SCENE II. A room in the Castle.

 Enter a Messenger.

MESSENGER.
Most noble sovereign, if he love that makes
His co
```

**prompt `'And he said, '`**

```
And he said, and he said so, yet shut her hip again?

GRATIANO.
Ay, your branches bear it further and mistress.

GRATIANO.
He shall not leave her with a stranger too.

PROVOST.
What means your worship that I can do?

GRATIANO.
Nor I, my lord, she’s as a gentleman as I have known to you.

DON PEDRO.
To use her to strike a stranger, and to say anything is the place
of an one. I will be distracted, and must be sa
```

**prompt `'The world is '`**

```
The world is desolate, speak here,
And hear me an old niece may profess
The pity of nature. I shall marry her,
And with the comfort of dark desert strong
That deserved the enemy guilty of her,
And leave her soldiers and cowards the field.

GREMIO.
And so I could, my lord.

KATHERINA.
I will follow my lord.

PETRUCHIO.
And, then, by this action, these fiends of the male
Define in my soul here upon my point.

KA
```

**prompt `'I '`**

```
I would they shall do accuse my father,
That they are constant. Which beginning together,
Although my clogs pass to hear a man to his letter,
I would open him well enough already.
I do beseech you, and I am ready for’t.

 [_Exit Servingman._]

HAMLET.
I have given him that lady and his subjects born of it. Good
you both here, you know your Grace must be ready. But it is a good astrange
with your goo
```

**prompt `'<free>'`**

```
[_Exit a Messenger._]

MESSENGER.
[_Reads_.] _He dies, and mine own soul I have entreated
On my age’s hate and he lies, which they have a doublet
That shall lead to your behalf. If he die,
In his father, he dies not.

MESSENGER.
Did I not lay thee for thy daughter,
Nor he hath not forgot the sin? Do see thee, Messala.

 [_Exeunt with Shylock and Shylock._]

SHYLOCK.
Thou shalt tarry the floorf a star,
And not true horses, being holy stronger,
Or in thy chair of my strong is a spacious business,
Or calls itself with thee and true-footed ducats,
At whose happy words should be horsed to death
And all thy bad sons with pity.

PORTER.
O Caesar!

NERISSA.
Is he not so much as much as he was in all
As he abjured now? Who was he as for his wife?

BRUTUS.
For he comes lodging to show him.

CAESAR.
```

### shakespeare_seed7

**prompt `'God '`**

```
God thou hast obey the cause of the Cardinal!

BRUTUS.
My cause was a tailor that his brother,
And though for his policy. Come, Cassius,
Taste and Messala, like a baby man,
Of many brothers’ conqueries and honours
Are all gentlemen.

BRUTUS.
Then do you good fellows?

CASSIUS.
Do you mean of that?

CASSIUS.
I am afeard that, sir. What tells?

CASSIUS.
What hast thou done ways sad?

CASSIUS.
I have the
```

**prompt `'The sea '`**

```
The sea for an uproar great man, the time
O’er wallow’d with lord; therefore I did see
The humble pavilions of this man’s world,
And he is come to hear this house.

SLY.
Here, ho! master, when I will make it only.

SLY.
To a house, master, I found them such a merry shoulders, and God innocent, I cannot
be made with me. When I can read a good thing I would not charge thee
a cause.

SLY.
Can that I be seen
```

**prompt `'Death '`**

```
Death the is as angel itself, are as pilligents as it says. Is not so? Madam?

PAGE.
I pray you, sir, is it not? Is not there?

SHALLOW.
Why, indeed, sir, hearty hath not intended. Dost thou live at my letter?

PAGE.
No, I pray. You could have no more words to say it well.

PAGE.
Why, I will not hear you.

SLENDER.
Fly, I will marry you the rest, I am your name Page.

PAGE.
Go to; and hear them hither,
```

**prompt `'And he said, '`**

```
And he said, craves the dead and news in France.
I know the devil, for the noble Duke shall be
A bottle with him.

ANNE.
It shall be a very man
In the season of a world’s brother,
And brought to proceed a pair of his own face.

 Enter His Servant.

His Servant, Master Servant, whither I come,
His grave, servants, and servants and soldiers.

HELENA.
I owe the gentleman for a bloody service,
To proceed with cond
```

**prompt `'The world is '`**

```
The world is best true; it is a man
As I told a man if it had not made
But will answer them, and detested to die.

ABHORSON.
I will here with my tempest; and if they can wish
Back me in their master’s judgment, I will not make
To make die with my arms to this right.

DUKE FREDERICK.
By heaven, I hate them too.

PRINCE EDWARD.
Pray you, you could not say you did.

[_Exit Duke Frederick with the bride._]

What s
```

**prompt `'I '`**

```
I will answer to this record.

 [_Exeunt Jaques and Claudio and Lords._]

CLAUDIO.
The chair of your own daughter’s a word believe
The next reforming lord of your lips upon
That compos’d forth to be angry of mine.
The Prince hath no reason lift that I can tell;
And then do I say, for I dare not marry
“By yea die, I know this plain consent kind of stealing.”
This plaint is not the first and great hol
```

**prompt `'<free>'`**

```
D FREDERICK.
So shall I, I will, and then I will make a man, the self shall
kill a dishes; so I will as shall say so. Bring it to the door.

BARDOLPH.
Sir, I will drink you when you are a man. I will but you go with your bonds.

PISTOL.
Go, you receive signifies not to come with him.

FALSTAFF.
No, no, you go not without Britain, not with a soldier’s cup of whether
particular knave you will. In France, I do beseech your highness.

PISTOL.
Re-enter Francis! O God, shepherd looks!

FRANCIS.
I have no friends to hear them the Duke of Brittany.

BOYET.
Good morrow, this is the matter; be mine, she that knows me
have anything to live. But for that she is, I will do love her to point me.
She is the state, and I brought her in shepherd’s head, she
loves her; shall we do it not. She would not be f
```


## Figures

- `fig_cross_bpc.png`
- `fig_eff_rank.png`
- `fig_head_layout.png`
- `fig_induction.png`
- `fig_resid_positional.png`
- `fig_spectra.png`