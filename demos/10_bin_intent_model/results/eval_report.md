# eval (calibrated)

2997 sentences

| | intent | feature | quality | box | count | **all five** |
| --- | --- | --- | --- | --- | --- | --- |
| all (2997) | 99.5% | 99.9% | 99.8% | 99.9% | 99.3% | **98.6%** |
| en (999) | 99.5% | 99.9% | 99.6% | 99.9% | 98.3% | **97.6%** |
| es (999) | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | **99.6%** |
| eu (999) | 99.2% | 99.9% | 99.8% | 99.8% | 99.6% | **98.6%** |

## Is the percentage right?

| question | accuracy | mean confidence | calibration error (ECE) |
| --- | --- | --- | --- |
| intent | 99.5% | 98.9% | 0.007 |
| feature | 99.9% | 99.9% | 0.001 |
| quality | 99.8% | 98.9% | 0.009 |
| box | 99.9% | 99.5% | 0.004 |
| count | 99.3% | 98.9% | 0.004 |

Refusing the answers the model is least sure of (confidence = weakest question that matters):

| confidence at least | sentences kept | all five right among them |
| --- | --- | --- |
| 0% | 100.0% | 98.6% |
| 50% | 98.1% | 99.5% |
| 70% | 97.5% | 99.8% |
| 80% | 97.3% | 99.9% |
| 90% | 96.6% | 100.0% |
| 95% | 95.9% | 100.0% |
| 99% | 85.3% | 100.0% |

## The two boxes

947 orders named the big or the white box. Sent to the wrong one of the two: **0** (0.00%). Read as another box or none: 1 (0.11%).

## Safety

- Sentences that must not move the robot read as an order: **0 of 912** (0.00%)
- stop recognised: 105 of 105

## Per intent (recognised)

| intent | recognised |
| --- | --- |
| ask | 99.4% (165) |
| both | 100.0% (225) |
| cancel | 98.5% (66) |
| chat | 99.4% (165) |
| hello | 96.0% (75) |
| move | 99.9% (780) |
| no | 100.0% (75) |
| pick | 100.0% (195) |
| refuse | 98.3% (120) |
| repeat | 100.0% (66) |
| reply | 98.7% (225) |
| sort | 99.8% (660) |
| stop | 100.0% (105) |
| yes | 98.7% (75) |

## Errors (42), first 40

- `eu` 'Jarri katua orain': intent: refuse -> chat (65%)
- `en` 'take the pieces that are bad': intent: move -> pick (77%)
- `eu` 'Zer moduz zaude?': intent: hello -> ask (83%)
- `en` 'can you please store the ones that have bolts into that big box': count: two -> all (43%)
- `en` 'group the ones that are bad in the white-colored box.': count: three -> all (41%)
- `eu` 'Gaizzki dauden eta pernorik ez daukaten bi pieza': quality: bad -> any (84%), box: none -> other (42%)
- `eu` 'Dena': count: all -> none (37%)
- `en` 'load the ones with a bolt on them insde this big box': count: two -> all (42%)
- `en` 'do nothiig': intent: refuse -> no (48%)
- `en` "let's set apart the bad ones fitted with bolts into the white box": count: two -> all (54%)
- `eu` 'orain dena lehenbailehen': intent: reply -> yes (23%), count: all -> none (70%)
- `eu` 'Pieza ona: bnaandu ezazu zurian.': intent: sort -> move (38%)
- `en` 'Go ahead and file the bad ones.': count: three -> all (49%)
- `en` "it's time to distribute the ones missing the bolt into the little box now": quality: bad -> any (49%), count: two -> all (43%)
- `en` 'isnide that white-colored box': intent: reply -> refuse (44%), box: white -> none (94%)
- `eu` 'Zer moduz?': intent: hello -> ask (45%)
- `en` 'organize the perfect ones that carry a bolt in the white-colored one, i mean to the big box': count: three -> all (45%)
- `eu` 'zenbat da bi gehi bi?': intent: chat -> ask (87%)
- `en` 'would you split the incorrect ones into this big box': count: two -> all (48%)
- `es` 'salduos': intent: hello -> stop (19%)
- `en` 'That one': intent: yes -> reply (53%)
- `en` 'pick up the ones that carry a bolt': quality: bad -> any (52%)
- `en` 'okay, group all of the broken pieces msising the bolt': feature: no bolt -> bolt (99%)
- `en` 'separate the correct ones with a bolt in the bigger box, sorry, into the white box': count: three -> all (45%)
- `en` 'alright, take the broken ones to that white-colored crate': count: two -> all (41%)
- `eu` 'hori': count: one -> none (51%)
- `eu` 'ondo, aktsik gabeko eta pernorik ez duten elementuak hartu itzazu': quality: good -> any (70%)
- `en` 'Group the ones fitted with bolts to that big box.': count: two -> all (44%)
- `en` 'take the ones fitted with bolts in that big one': count: three -> all (41%)
- `en` 'Hey robot, could you file the ones missing the bolt in the box on the left?': quality: good -> any (48%), count: two -> all (44%)
- `es` 'oye robot, ordena las piezas con caidado': feature: any -> no bolt (56%), quality: any -> bad (60%)
- `eu` 'amaitu duzu?': intent: ask -> stop (47%)
- `eu` 'kutxa hadienera perno gabe dauden objektuak sartu': box: big -> other (65%)
- `en` 'Please, please send the ones that have bolts in that bigger box': quality: bad -> any (50%), count: three -> all (43%)
- `es` 'Esa rápido': count: one -> none (50%)
- `eu` 'hori orain': intent: reply -> yes (29%), count: one -> none (59%)
- `es` 'La planca': box: white -> none (45%)
- `en` "it's time to sort the ones without a bolt in this white one": count: two -> all (43%)
- `en` 'please deposit the fine ones that have a bolt to the larger box': count: three -> all (46%)
- `eu` 'hartu perno gbaeko akastunak.': feature: no bolt -> bolt (67%)

## Without calibration (temperature 1)

| question | ECE before | ECE after |
| --- | --- | --- |
| intent | 0.004 | 0.007 |
| feature | 0.001 | 0.001 |
| quality | 0.002 | 0.009 |
| box | 0.001 | 0.004 |
| count | 0.005 | 0.004 |
