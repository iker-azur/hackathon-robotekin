# real (calibrated)

143 sentences

| | intent | feature | quality | box | count | **all five** |
| --- | --- | --- | --- | --- | --- | --- |
| all (143) | 98.6% | 98.6% | 100.0% | 100.0% | 97.9% | **97.9%** |
| en (49) | 98.0% | 98.0% | 100.0% | 100.0% | 98.0% | **98.0%** |
| es (49) | 98.0% | 98.0% | 100.0% | 100.0% | 95.9% | **95.9%** |
| eu (45) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |

## Is the percentage right?

| question | accuracy | mean confidence | calibration error (ECE) |
| --- | --- | --- | --- |
| intent | 98.6% | 98.9% | 0.011 |
| feature | 98.6% | 99.9% | 0.013 |
| quality | 100.0% | 98.6% | 0.014 |
| box | 100.0% | 99.5% | 0.005 |
| count | 97.9% | 98.2% | 0.028 |

Refusing the answers the model is least sure of (confidence = weakest question that matters):

| confidence at least | sentences kept | all five right among them |
| --- | --- | --- |
| 0% | 100.0% | 97.9% |
| 50% | 98.6% | 97.9% |
| 70% | 97.9% | 98.6% |
| 80% | 97.9% | 98.6% |
| 90% | 97.2% | 98.6% |
| 95% | 97.2% | 98.6% |
| 99% | 74.1% | 98.1% |

## The two boxes

54 orders named the big or the white box. Sent to the wrong one of the two: **0** (0.00%). Read as another box or none: 0 (0.00%).

## Safety

- Sentences that must not move the robot read as an order: **0 of 56** (0.00%)
- stop recognised: 6 of 6

## Per intent (recognised)

| intent | recognised |
| --- | --- |
| ask | 100.0% (6) |
| both | 100.0% (3) |
| cancel | 100.0% (6) |
| chat | 100.0% (6) |
| hello | 100.0% (9) |
| move | 100.0% (50) |
| no | 100.0% (3) |
| pick | 100.0% (6) |
| refuse | 75.0% (8) |
| repeat | 100.0% (6) |
| reply | 100.0% (15) |
| sort | 100.0% (13) |
| stop | 100.0% (6) |
| yes | 100.0% (6) |

## Errors (3), first 40

- `es` 'Sí, esa': count: none -> one (86%)
- `es` 'Estas piezas no tienen perno': intent: refuse -> reply (99%), feature: any -> no bolt (100%), count: none -> all (100%)
- `en` 'These pieces have no bolt': intent: refuse -> reply (99%), feature: any -> no bolt (100%), count: none -> all (100%)

## Without calibration (temperature 1)

| question | ECE before | ECE after |
| --- | --- | --- |
| intent | 0.014 | 0.011 |
| feature | 0.014 | 0.013 |
| quality | 0.000 | 0.014 |
| box | 0.000 | 0.005 |
| count | 0.025 | 0.028 |
