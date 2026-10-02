# Persona transformers — what the weights say

Models: mccarthy, mccarthy_excerpts, melville, melville_v1, shakespeare, shakespeare_seed7. Each is a 6-layer, 6-head, 384-wide character GPT (nanoGPT shakespeare_char config), trained from scratch on that author alone, shared 92-character vocabulary.

## Training

| model | best val loss (nats/char) | bits/char | at iter | train chars |
|---|---:|---:|---:|---:|
| mccarthy | 1.052 | 1.518 | 4750 | 5.23M |
| mccarthy_excerpts | 1.274 | 1.837 | 2300 | 0.60M |
| melville | 1.230 | 1.775 | 5000 | 8.09M |
| melville_v1 | 1.211 | 1.747 | 5000 | 8.09M |
| shakespeare | 1.327 | 1.915 | 4750 | 4.82M |
| shakespeare_seed7 | 1.329 | 1.917 | 5000 | 4.82M |

## Cross-perplexity: how each model reads each author

Rows are models, columns are held-out text, cells are bits per character (lower = more predictable to that model).

| | mccarthy text | mccarthy_excerpts text | melville text | melville_v1 text | shakespeare text | shakespeare_seed7 text |
|---|---:|---:|---:|---:|---:|---:|
| **mccarthy model** | 1.470 | 1.742 | 2.553 | 2.553 | 3.117 | 3.117 |
| **mccarthy_excerpts model** | 1.965 | 1.838 | 3.429 | 3.429 | 3.801 | 3.801 |
| **melville model** | 2.735 | 3.008 | 1.851 | 1.851 | 2.239 | 2.239 |
| **melville_v1 model** | 2.738 | 3.047 | 1.861 | 1.861 | 2.299 | 2.299 |
| **shakespeare model** | 2.568 | 2.732 | 2.394 | 2.394 | 1.868 | 1.868 |
| **shakespeare_seed7 model** | 2.800 | 3.043 | 2.400 | 2.400 | 1.873 | 1.873 |

### Probe texts no model trained on

Bits per character on held-out prose (lower = the model finds it more natural).

| model | mccarthy_essays |
|---|---:|
| **mccarthy model** | 2.288 |
| **mccarthy_excerpts model** | 2.563 |
| **melville model** | 3.125 |
| **melville_v1 model** | 3.172 |
| **shakespeare model** | 3.231 |
| **shakespeare_seed7 model** | 3.522 |

## Attention-head layout (mean over held-out windows)

**mccarthy** — mean normalised entropy 0.531; mean attention distance 19.7 chars (layer means 4.3, 33.2, 8.6, 13.9, 22.5, 35.6); 3 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H3 = 0.01; heads with induction > 0.1: 0.

**mccarthy_excerpts** — mean normalised entropy 0.537; mean attention distance 19.5 chars (layer means 2.8, 44.0, 9.3, 14.4, 19.4, 26.9); 6 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H1 = 0.01; heads with induction > 0.1: 0.

**melville** — mean normalised entropy 0.477; mean attention distance 18.2 chars (layer means 5.0, 34.9, 5.6, 21.2, 17.8, 24.9); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L1H2 = 0.01; heads with induction > 0.1: 0.

**melville_v1** — mean normalised entropy 0.480; mean attention distance 20.3 chars (layer means 4.5, 44.6, 8.8, 9.2, 23.2, 31.6); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H0 = 0.01; heads with induction > 0.1: 0.

**shakespeare** — mean normalised entropy 0.491; mean attention distance 21.4 chars (layer means 3.6, 50.0, 6.7, 9.4, 25.9, 32.5); 5 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L4H4 = 0.08; heads with induction > 0.1: 0.

**shakespeare_seed7** — mean normalised entropy 0.509; mean attention distance 22.9 chars (layer means 4.2, 53.7, 6.0, 9.2, 19.1, 45.1); 4 previous-token heads (>0.5 mass); 0 first-token-sink heads; strongest induction head L5H4 = 0.04; heads with induction > 0.1: 0.

## Spectra of the learned operators

Effective rank = exp(entropy of the normalised squared singular values): how many directions the operator really uses. Decay exponent = slope of log σᵢ vs log i over the top half of the spectrum (more negative = the operator is dominated by a few directions).

| model | QK eff. rank (of 64) | OV eff. rank (of 64) | MLP-in eff. rank (of 384) | wte eff. rank (of 92) | QK decay | OV decay | MLP decay |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 20.4 | 24.8 | 152.1 | 28.5 | -6.32 | -6.31 | -0.46 |
| mccarthy_excerpts | 19.9 | 26.4 | 139.2 | 18.9 | -6.33 | -6.33 | -0.49 |
| melville | 20.5 | 26.0 | 147.1 | 33.5 | -6.32 | -6.31 | -0.47 |
| melville_v1 | 20.2 | 26.0 | 146.4 | 34.2 | -6.32 | -6.31 | -0.47 |
| shakespeare | 20.9 | 27.3 | 147.6 | 30.8 | -6.33 | -6.32 | -0.47 |
| shakespeare_seed7 | 21.8 | 27.4 | 148.2 | 31.3 | -6.32 | -6.32 | -0.46 |

## Martin–Mahoney alpha (power-law tail of the eigenvalue spectrum of WᵀW)

Near 2 = well-trained layer; below 2 = over-trained; above 6 = random. Mean over layers.

| model | Wq | Wk | Wv | Wo | mlp_in | mlp_out |
|---|---:|---:|---:|---:|---:|---:|
| mccarthy | 2.13 | 2.51 | 2.19 | 2.19 | 1.71 | 1.98 |
| mccarthy_excerpts | 1.89 | 1.95 | 2.21 | 1.88 | 1.84 | 1.81 |
| melville | 2.08 | 2.32 | 2.20 | 2.41 | 1.71 | 1.99 |
| melville_v1 | 1.89 | 2.13 | 2.23 | 2.21 | 1.71 | 2.00 |
| shakespeare | 2.12 | 2.31 | 1.84 | 2.07 | 1.70 | 2.13 |
| shakespeare_seed7 | 1.91 | 2.18 | 2.19 | 2.01 | 1.71 | 2.17 |

## Logit lens: bits/char if the model stopped after block k

| model | embed | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mccarthy | 25.50 | 3.88 | 2.98 | 2.47 | 1.96 | 1.62 | 1.51 |
| mccarthy_excerpts | 24.57 | 3.92 | 3.25 | 2.82 | 2.42 | 2.12 | 1.98 |
| melville | 24.65 | 4.16 | 3.44 | 3.02 | 2.59 | 2.16 | 1.92 |
| melville_v1 | 25.22 | 4.14 | 3.49 | 3.04 | 2.60 | 2.16 | 1.93 |
| shakespeare | 25.20 | 3.98 | 3.32 | 2.91 | 2.47 | 2.04 | 1.90 |
| shakespeare_seed7 | 25.93 | 4.02 | 3.33 | 2.90 | 2.46 | 2.07 | 1.90 |

## Residual stream and position

**mccarthy** — residual norm by block: 1.5, 13.1, 20.8, 28.1, 35.0, 42.9, 52.4; attention:MLP update ratio per block: 0.44, 0.62, 0.78, 0.67, 0.51, 0.36; position-embedding spectral centroid 29.8 cycles/window, 23% of power below 8 cycles.

**mccarthy_excerpts** — residual norm by block: 1.3, 10.6, 15.9, 21.4, 26.2, 30.5, 34.6; attention:MLP update ratio per block: 0.42, 0.56, 0.90, 0.74, 0.63, 0.46; position-embedding spectral centroid 35.6 cycles/window, 19% of power below 8 cycles.

**melville** — residual norm by block: 1.5, 13.1, 21.2, 28.7, 34.9, 42.2, 52.4; attention:MLP update ratio per block: 0.37, 0.64, 0.78, 0.69, 0.59, 0.41; position-embedding spectral centroid 30.7 cycles/window, 22% of power below 8 cycles.

**melville_v1** — residual norm by block: 1.5, 13.2, 20.5, 27.6, 35.0, 43.0, 52.4; attention:MLP update ratio per block: 0.36, 0.62, 0.76, 0.70, 0.55, 0.41; position-embedding spectral centroid 30.0 cycles/window, 29% of power below 8 cycles.

**shakespeare** — residual norm by block: 1.5, 12.9, 19.6, 26.9, 34.5, 42.3, 52.5; attention:MLP update ratio per block: 0.38, 0.59, 0.82, 0.69, 0.58, 0.41; position-embedding spectral centroid 32.5 cycles/window, 24% of power below 8 cycles.

**shakespeare_seed7** — residual norm by block: 1.5, 12.7, 19.2, 26.4, 33.6, 42.4, 52.3; attention:MLP update ratio per block: 0.39, 0.59, 0.85, 0.69, 0.53, 0.40; position-embedding spectral centroid 31.0 cycles/window, 25% of power below 8 cycles.

## Worldview probe: the first word each model puts after a shared prompt

96 sampled continuations per prompt (temperature 0.9, top-k 40); counts of the first word.

**`God is`**

- mccarthy: that (12), it (8), a (6), he (5), the (5), what (5), not (3), an (3)
- mccarthy_excerpts: that (11), a (9), not (7), just (7), no (6), to (4), goin (3), what (3)
- melville: a (11), not (9), the (7), in (4), good (3), all (3), an (3), this (2)
- melville_v1: the (14), a (10), not (7), all (4), an (2), now (2), good (2), only (2)
- shakespeare: not (16), a (6), as (4), he (3), no (3), your (3), but (3), the (2)
- shakespeare_seed7: not (10), a (9), the (6), so (5), it (5), good (4), his (3), as (3)

**`Death is`**

- mccarthy: a (20), that (8), all (6), the (5), it (4), he (4), not (2), cold (2)
- mccarthy_excerpts: not (16), no (14), a (11), that (11), the (6), to (3), at (3), true (2)
- melville: the (13), a (10), not (8), it (3), but (3), an (2), to (2), all (2)
- melville_v1: the (15), a (13), not (11), so (2), something (2), in (2), only (2), hereabout (1)
- shakespeare: not (11), the (8), dead (6), a (5), no (4), to (4), too (3), but (3)
- shakespeare_seed7: not (10), the (6), an (5), a (5), my (4), but (3), as (3), at (2)

**`The world is`**

- mccarthy: not (15), the (10), a (9), what (5), no (4), that (4), it (3), being (2)
- mccarthy_excerpts: in (14), not (13), a (8), made (6), that (5), no (5), the (4), for (2)
- melville: not (10), a (10), the (7), no (5), to (3), so (3), as (2), then (2)
- melville_v1: a (10), the (10), not (10), but (3), never (2), good (2), of (2), absolutely (2)
- shakespeare: the (9), a (8), not (7), dead (5), no (4), my (3), all (3), comes (2)
- shakespeare_seed7: not (13), a (5), but (4), at (4), all (4), my (3), in (3), gone (3)

**`A man is`**

- mccarthy: a (13), not (6), in (6), just (4), the (4), going (3), goin (2), watching (2)
- mccarthy_excerpts: a (16), not (12), that (6), the (5), no (5), it (4), what (3), like (2)
- melville: a (12), not (9), the (6), in (4), to (3), more (2), made (2), no (2)
- melville_v1: the (11), a (9), his (6), not (6), no (3), to (3), gone (2), all (2)
- shakespeare: a (14), the (9), not (6), come (4), no (3), in (3), so (3), almost (2)
- shakespeare_seed7: a (14), the (11), not (7), my (3), made (2), no (2), his (2), to (2)

**`Love is`**

- mccarthy: a (12), that (7), not (7), the (7), it (4), just (3), all (3), you (2)
- mccarthy_excerpts: the (23), no (11), a (10), that (10), not (7), to (4), an (2), such (2)
- melville: a (11), the (7), in (4), made (3), not (3), to (2), more (2), but (2)
- melville_v1: the (12), a (10), not (7), more (3), in (2), gone (2), all (2), no (2)
- shakespeare: a (12), not (10), the (7), dead (4), our (2), an (2), coming (2), too (2)
- shakespeare_seed7: a (8), not (8), the (8), as (4), my (4), very (4), thy (3), too (3)

**`The sea is`**

- mccarthy: a (8), in (6), the (6), what (5), not (4), gone (4), still (3), going (2)
- mccarthy_excerpts: not (23), no (22), the (10), that (6), to (4), a (4), some (1), of (1)
- melville: a (16), not (8), the (5), but (3), at (3), no (2), one (2), to (2)
- melville_v1: a (12), not (10), the (8), no (3), an (3), his (3), but (2), now (2)
- shakespeare: a (9), the (8), not (7), gone (3), in (3), but (3), come (2), as (2)
- shakespeare_seed7: the (9), a (4), as (3), his (3), our (3), most (3), not (3), on (2)

**`There is no`**

- mccarthy: sound (5), longer (4), sign (4), one (3), part (3), great (2), such (2), idea (2)
- mccarthy_excerpts: way (12), one (8), god (7), such (4), words (2), idea (2), longer (2), bottom (2)
- melville: more (8), means (2), stranger (2), small (2), light (2), little (2), doubt (2), terrible (2)
- melville_v1: more (14), means (4), one (4), very (3), doubt (3), longer (3), end (2), other (2)
- shakespeare: more (13), matter (7), less (6), man (5), friend (3), better (3), such (3), noise (2)
- shakespeare_seed7: more (21), man (6), such (5), matter (4), further (3), mater (2), very (2), cause (2)

**`Man is`**

- mccarthy: a (18), the (6), not (4), dead (4), so (3), that (3), about (3), no (3)
- mccarthy_excerpts: that (16), the (10), not (9), it (6), what (4), just (3), a (3), forgiveness (2)
- melville: the (11), a (10), not (5), an (4), no (4), in (3), for (2), but (2)
- melville_v1: the (14), a (11), not (9), no (5), in (3), gone (3), one (2), this (2)
- shakespeare: a (11), not (8), the (6), in (4), my (4), that (3), dead (3), too (3)
- shakespeare_seed7: a (13), the (8), in (3), so (3), something (3), not (3), like (2), at (2)

## Samples (temperature 0.8, top-k 40)

### mccarthy

**prompt `'God '`**

```
God it aint nothin to see it. And I was not much used to be their troubles. I got to take you out there for a long time to be six here in the courthouse is probably a thing and that means they dont know what it’s a lot you wanted to hear, he said. I seen this country up, and not more ago away at it.

How long you all do you know what you’re talkin about this lawyer?

What’s she done in the heelbox?

I
```

**prompt `'The sea '`**

```
The sea that creaked like one of the rights fanced still with their clothes. Beyond the creosote walls the willows striped along the floor along willows on each and he sat there with his hands crossed behind him and looked around. John Grady put his hat on the pocket and looked at the guards.

In the sun he met them hot days and fell away on the street. The man had turned to the horses and he looked back
```

**prompt `'Death '`**

```
Death for seventy-five days. They’re get a little of horses out of the army with daughter.

What old man said were talking about him?

The juggler swung his hat out into the formall and the moon spat and squatted looking at the bar and then bent and stood looking out across the road, the boy looking down at it and then then the woman blew, the other one hand on her side and her tongue. She said: Hold it
```

**prompt `'And he said, '`**

```
And he said, were stilled and muttering and he could see cautiously beyond the white clay and he would have been halfway to show the money and he said that he was his brother but he did not shoot him.

He looked up at the horse while he looked at the rider spoke to himself and then rode up and went back down the street and down toward the street toward the dogs and out toward the railroad toward the driver. Th
```

**prompt `'The world is '`**

```
The world is not a only violence. An older man’s revolution and here he can see anybody be upon.

Yes. A man was going to hear the willing of nonether. Or maybe you would just see him.

That you had one of the girls to decide him to receive his side.

Why are the people at?

Mr Owens.

The fuck is not the second case of strangers.

He dont know. I dont know what it is.

He said that it was not much that they w
```

**prompt `'I '`**

```
I wasnt no more.

I wouldnt do it.

Boyd pushed back the chair on the bed and put his hand on the side of the stove.

Would you talk to him about him?

What happened to you?

I dont know. I aint seen him.

I aint nobody knowed it.

Well.

Here. He looked at the man again. He looked at the blind man that he had a child man a gutter to find. I told you’d get his shoes in the dead way was you in mind.
```

**prompt `'<free>'`**

```
They were realization with how it was that which it was a job of life in the mornings there was a man who wore it was a boy. So that they would have no death for its father to have any part of it. Out on the sea a void of valt land on the hallway to the rear of the universe. A floor of dust among the horses bright and the water stood in the grass and dust lay in the dead country beyond the night and the sky singled for the white dry stoical darkness. There were no barber backs and broken by cart below lung out of the poles before them holding the house and he talked to his goodness, a boy in his early such and taking up the street.

He aint been mind to come, he said.

They rode out to the side of the burners where it brought a parket of dust and rope and a three of them and the pigs cross
```

### mccarthy_excerpts

**prompt `'God '`**

```
God fear on the three hours. They aint nothin with my cold and silence and the space to the same weight to him. The road gray in the woods of the horse and some could hear the tanding for the terrible softs with the shorelights leaves and they rode on their millions in the wet and red all red among the dark and he had never left and drawn up and she woke on the dark and stroke and her and went came in
```

**prompt `'The sea '`**

```
The sea thing of the world is one. If I will see it. It is meant to everything else. The halt world was contained about it. It aint nothin to be able or some bready that it aint a fair if it we dont believe in the cath. The night stop come and set it was a child and I have no acces to be a certain points. And in that the world were gone on it. It was always despair. We have no still better it will be a li
```

**prompt `'Death '`**

```
Death to see it and every might have been has no time. But there is no more of breath is a true that is a forcing more less are made only the game is not a thing can be not a chorade of it. What's not the word is so but the way the world is made of the world which is thing is the world in the world is not so only but nothing is nothing in the world. But when he is he never knows to know what he said wha
```

**prompt `'And he said, '`**

```
And he said, she wasnt too as the bloods. He came to the leaves when he building it shifted and was in the water saddles and he'd carry as wall to be. Or there were signs of his more but rathers are no recognized to grape the witness of them until he could find see the world and that itself to strange the concept of his prospect of his will he is not end. His memory are not an acts if there no might have been
```

**prompt `'The world is '`**

```
The world is a good about in everything which is that which you stand to say. But when that is always what you want to be able to the world. It can be a world idea that event be always known to do. I don't know what you see it. It wouldnt get mean what is. There would be as always be forgiving. It is that you said that what is just that you've forgets the take and that you dont know what I dont know. You could
```

**prompt `'I '`**

```
I was my life. I dont know it will be one. That aint no one thing. Yessir. And it is pretty much a dream. Mach man is a made of it. The world is not come but the world is that is also and it is not make which mease is not some other others. No, I do. It's just sure what will tell you it is in that he's been known. And the one way that it would be the worst like this place and can never be for it. Th
```

**prompt `'<free>'`**

```
Lore pa?s contemplative is that harded such a person with the world and all that was.

The world was little convictions of the same which is not being to any time to be capable to discoverable the condition. And the beast senses lies within the testiny inction the winterms of her and the final light flames and she was cold. There were no more, the old man said that all of we no made. The world is gift to be the path of the town. The desert is not an one is not a thing idea inforce and the doubts that the best shadow of the world makes in the case in its sort of which is the world in missit of it and that also make a certain mathematics history is not inheritence as itself an its to be here.

The shores of looking in the mountains or cry and the truck souls that follow the world was on the
```

### melville

**prompt `'God '`**

```
God keep the
recurrence about that some ten of that sort of vault. What is the grand
subject of the city? The fare should not be the period, not a more
than the same words; some small head of the top could
start in the hands of the black long-days of the iron darkers, and
in facts of the rear of distances I came to be present. I had an iron word
that we can not have taken my heart and my mood; so I ha
```

**prompt `'The sea '`**

```
The sea went; he was a few moments to have seen
him, and I looked among my small countenances, and ere I had recoiled
the thing which I had not taken his turn, and then round him descending
from him to his employer. He was absolutely producing to the place of the
prowling of Paris, he was at the same word, and more rendered than,
that some night the mate was now coming on his glance. The main-top-man
was
```

**prompt `'Death '`**

```
Death the ship after the nobler, I was said to my
stump; and no portion; I ever saw the mere monotonizon, not the
others who all this men in the presence of the time. I know that there
is a course of sight he has a little sense of old tabered character;
nor else is it the unional false as in the same distance, or otherwise
estrange, in truth, would scarce such precise services of the devils
of maintaini
```

**prompt `'And he said, '`**

```
And he said, quite as he ran ye down;
  And presently declined with your branches behind.
     So bad might convince yet there sword
      Wnose and night, to the coin.

They make us for lake that nor the sun.

They shall see an end, and they show they say.

The same result might fain have their crews
       And strives often to the Almost weary--
                    A rock of course dipped in the
          Ad
```

**prompt `'The world is '`**

```
The world is desired to
desire the rest of the mere and each part of the temples.

Here the whispers draw nigh; the foot had started on, and caught the
most sort of the Surgeon, gliding into the floor, torn forth into a
cowerful hour there was a rush who secretly superstitiously built into
its head. For my estimations had made his concern as the crew, I was obliged to
recurre it. But it had but a carriage for
```

**prompt `'I '`**

```
I would not thought of the more fellows; and the last constrained;
and now instead of events as he had long passed round along the
well-distance of the enemy, were expressly to be sure. And laughed at that
crew, one of the other excellent scene, the heavy man lies of grave
laws, and camel-adaptible women.

CHAPTER LXXXII.
THE RANDOW OF THE GENERAL FASHION

A TENTH of the Terra Bibles of His Semigran
```

**prompt `'<free>'`**

```
on the surface.
The boat was returned towards the ship, and began to sail in the
contempt. But that had rendered a little paper to which the same few
Laying Alexandro, and shaking his head warriors, to be grounded to call
upon the chest, in an evening, though his person was impelled by reader and
broader, dibiously with the lump hull of his skulls. He had been minutely
crossing through the heart of a man, convinced his gallant brain lay over
the wind, where a circumstance, shaped, strong shadowy brow, and the
new pantiless articles of the pump-like places, the tradest wings were the
long-small barbarous whalers of lips and seamen. The effect had been
useless to remember them, in particular, was therefore, and in other
restrictions of the altar, feeling become aloft to some other things and
```

### melville_v1

**prompt `'God '`**

```
God thou art, oh! that can thou not be battle-holes
of thoughts bent it into the remorseless end; who who for the solitary
corner? What beast is the sea-far. No, come to me, Pierre."

"I can't not ate length to be in upon getting the first word, but too
late the stranger, that the whole same tale is being a terrific on the
last part of the first fact, but in the high concealed condition of the
prophet
```

**prompt `'The sea '`**

```
The sea done, but once more more than the most wall of the sun; while,
after close close, while the vessels of the horizontal figure has fallen
into the eastern sea.

“Dark, the precise will not with me, men?”

“What do you should all disclose as the mast?”

“Stiff the sailors succeed almost as his oars the prow?”

“Hark! half your years, now, that if entirely doctor you are the first
man to be hard.”

“W
```

**prompt `'Death '`**

```
Death and in a tall!—

How long, are pilling the chings of the frigate caused by the conduct
of a hour. Wherever the heart was the very sea-window in the deck,
each was thirty jaws of waves are of the Pontiffs, in a midst of watery. The
column, with a wonderful process of the last side, who, as if they really dropped
upon the spirit of the ships, that from their arms were seen to the
shoulder-still hear
```

**prompt `'And he said, '`**

```
And he said, 'In the meads, that our directions said it that, the
author of the summer, thou wiltest for some other more than passage, in
the middle of this full sight of the Black Neversink was precious at the
Pacific. The Chaplain further in the San Morai had been tried upon the
transferred portions of the descent and not more than the measure of
those things that had his been no doubt to set his deliberate
```

**prompt `'The world is '`**

```
The world is the thresholding, and the patients of his invalid
wife; one hath been arrived from a still watery glimpse of his arm,
and in less than the works and least than for the clear coast with a
man-of-war’s-man, in a fiction of a Cepher fell to pump his respect, and the
announcement of the blacks, when the man glided over himself to a probability
with a judging of purposely resemblance, and was pricking
```

**prompt `'I '`**

```
I will answer to this reception.

I am but to say Somewhere who has seen the whale’s hand, and always been
drawn, when the boat stood like a Pip, in the sea, both lighted in the
coast, the least did an honest door like a husband’s grace would be truly called
to him, which could do it not but that he had been distinguished in the
mate which of the stranger had been intended by the forecastle of the
s
```

**prompt `'<free>'`**

```
Daring the captain concerning the Navy himself atmosphere, as if such a
terrible attended to the ship, over which some disinterested form he
had not wanted to return the skirts to the table. A native of the conduct
we could not be merited why so with a singular to land unwash the
further of the vapors of the shadowy grounds of the deck.

Had the captain of the tree particular to the circumstance of the Serapis,
doubtless in the same landsman-of-war’s-mates and size of the long
times before the boat. And the starboard was often planking to the tempest
of the sea. But to this excuse of the crowd that stuck was a view of puff,
which in a chain generally called upon this space of a mariner, we
were at once pursued that this steward like a rope, the captain was seen the
continual of a barge, wh
```

### shakespeare

**prompt `'God '`**

```
God knows he that could draw this town? Your country’s a
skild out of her winds; if you would come upon him, your affection to you
said so, for I would have anything before you would live in Coventry.

FIRST CLOWN.
That same way he hath misused her; but some whose contents is on his
words and his wit hath borne his vice, and he needs not astone if he
has come from his own men come to drink him hither.
```

**prompt `'The sea '`**

```
The sea ends with the valiant time his virtue:
How far the world does the war, looks on bed,
Here sits the world worse in the bosoms of entertainment
Of nature or shall seem out. So please you, to fight.

HAMLET.
[_Aside to Rosencrantz._] How does the King in this contrary?

HAMLET.
There is all the rest and most fair service.

HAMLET.
Spoke by me, my lord, and thog more.

HAMLET.
No, no, no; but, my lord
```

**prompt `'Death '`**

```
Death that they have strongly love so far in the morning shall be
such a richer’s dangerous cheek, nor would have the presage of the war, have prevented
in the high and maiden voice of great and place and die at his
mind and answer: and here he lies for the property of a courtier, a good
courteous lady. Here the Duke will send the fly in me in me; here is a good
breath to any doath; I will show the peer
```

**prompt `'And he said, '`**

```
And he said, though not prepare me half this like world
me he is not.

PANDARUS.
My nature, my lord.

CRESSIDA.
Flesh and proud French thy life!

PANDARUS.
No, for thou shalt speak in pardon.

PANDARUS.
’Tis pardon of thy time; for thou art no content of thy friend
worse.

CRESSIDA.
I hope thou art a condemned time for the wanton of men, and so proudly
so proud as thou art.

PANDARUS.
Why, we will.

CRESSIDA.
```

**prompt `'The world is '`**

```
The world is a word that he lies a day, and in his word
touches to him off. Therefore look not on him in him, I will weep him like a
woman with a lie in a back craft. It is not a fairer that I may not lie.

PAROLLES.
Sir, I have not enough an end of being on years. Your fair fair powers will linen
when I mean to pardon you, we’ll pray the nose of a day, and I ever
seal the for a fair complaint of my father’s l
```

**prompt `'I '`**

```
I pray you, that I will look for you for me.

TIMON.
Well, fare you well.

TIMON.
Why are you speak straight in heaven, that our own men?

APEMANTUS.
Our authority is passed.

TIMON.
I do beseech your hand to charge me.

TIMON.
Then we mean the cause of man robes our rests
Within this blast truth with the poetry of heaven:
For us to this is this to stabs the cannon;
But then my soldiers’ sake is sti
```

**prompt `'<free>'`**

```
I have not seen him, nor him follow’d.

PANDARUS.
More cherish’d charm your plenteous red for this to prevent him.

PINDARUS.
I pray you, call the count of these fingers ways.

PANDARUS.
What canst thou do? What ears? Why, shall I have?

PANDARUS.
No, sir, I have not spoke in you.

PRINCE.
I will not stop the mind him.

PANDARUS.
Let us go from me.

PRINCE.
I confess thee, sir, and put it so.

PRINCE.
Good Monsieur Boys.

Enter Boy, Rosalind.

BOY.
Good Monsieur Boy, you mean to me from the curse of it.

ROSALIND.
For the judgement of our sisters are no curses the playment,
The gave way here he by malignanims and dishonour are
to hear his heart, and watch him well.

CELIA.
And you, my lord.

ROSALIND.
What, what a play?

CELIA.
I have done the world recovers to be the world to be noted to
```

### shakespeare_seed7

**prompt `'God '`**

```
God then, what thou say’st that keep out the dog
Of thy beauty, the fall of Lancaster, thou hast been.
I wassured and fortune to move, that in thy life
I never entreat thy father came.

[_Exeunt._]

SCENE III. Another part of the field, give me him a chamber of the
Beautier. An Aphmantus’ tent.

Enter Aphlonio, Prolonius, Lucius, Chatillion and Sebastian.

APEMANTUS.
My mistress, and here is my joy to
```

**prompt `'The sea '`**

```
The sea new among the devil’s nurse
That she had been blest as her breath as she desires
Of greatness but approached into a sleep
And in thick and discoursed bloody place.

BASTARD.
Well, then, call me hence; I am out of her.

KING.
I see you that I had this fellow. He’s so much undertaken.

BASTARD.
By this same having yet, and ’tis true;
When I will add th’ arbitrement with her friend,
With most riderio
```

**prompt `'Death '`**

```
Death he hath cause.

AGAMEMNON.
To the drink about me, and so should I.
The day is come to the foot. The sum of victory
Looks steal with cursed and prove bought in the child
Of everlasting corner. The grief holds up the dead;
Put it and to the same five windows as and strength
To have me but this best of birth, or abhorr’d we,
Some banished body of the air, or his town,
Whose inheritances was argued to
```

**prompt `'And he said, '`**

```
And he said, and she calls and see him for some sanctuary, with
the most remembrance of a poor master’s graces, none but to quench. I am fair
traitor. Here, there is a pupposed man and a hundred earth. He had a fool, a
fool, a husband to be welcome a dream. There is a foolish that plagues out of
honourable plains, and for he hath made a good thing, that such a condition is
as good a good servant. Hume, let me
```

**prompt `'The world is '`**

```
The world is thy fondman, and double thy shame,
Bladed with thy statute, in the particular.

CAESAR.
Why, it is best possible, as thought hadst seen
As before thy soul should be rough; we will understand
My parting things to thee. I will bestow my soul,
And I can soon one but by a lamentable
That would do me what I should have thee for direction
To see them that which thou dost say they take it,
Which I am afr
```

**prompt `'I '`**

```
I will do. Let him go.

[_Exit, and Captain._]

This is the slave, and let me know why.
This is the time of the better painted men;
Sweet Montague, valiant carrion by love
And by the world, which, as it hath mov’d me.
It stands it in blood, as it stands on.

 Enter Captain.

CAPTAIN.
Why is he had, my Lady Montague Captain?

CAPTAIN.
All health should be heard and true health of a king.
He, I ask th
```

**prompt `'<free>'`**

```
THERSITES.
You are a good company.

THERSITES.
Truly, being my liege, a poor duke.

THERSITES.
As for a sport.

THERSITES.
The empress, when you are a peevish that you shall do it. You shall
understand them to sack.

THERSITES.
Then remember were to blow at some little business in your own
general.

THERSITES.
It is no matter which stood at the versal world; now for the next world were they
well did invent in any other sacrifice and the world alive in their
assembly as they should seem for me. But, farewell.—When women should be
honestly taken.

 [_Exit._]

SCENE III. Rome. A Room in Antony’s House.

 Enter Antony and Servant.

ANTONY.
If what is thy sickness are thy dear master?

SERVANT.
No, nothing so.

ANTONY.
Dead.

ANTONY.
So there were the labour in my behaviour.

ANTONY.
What say
```


## Figures

- `fig_cross_bpc.png`
- `fig_delta.png`
- `fig_eff_rank.png`
- `fig_head_layout.png`
- `fig_induction.png`
- `fig_ncd.png`
- `fig_resid_positional.png`
- `fig_spectra.png`