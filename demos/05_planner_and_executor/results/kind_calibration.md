| truth kind | blobs | area px | elongation | fill |
|---|---|---|---|---|
| ball | 40 | 354..391 | 0.922..0.998 | 0.9982..1.0000 |
| cube | 40 | 677..723 | 0.833..0.893 | 0.9570..0.9840 |

## where the threshold goes

Cubes top out at 0.9840 and balls start at 0.9982, so the two ranges do not touch. Any threshold in (0.9840, 0.9982) separates them; 0.9915 currently sits at 0.9915.
The middle of that gap is 0.9911.

## kind accuracy at the current thresholds

`BALL_FILL` = 0.9915, `BALL_ELONGATION` = 0.907

- correct: 80/80 (100.0%)
