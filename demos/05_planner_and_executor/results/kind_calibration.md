| truth kind | blobs | area px | elongation | fill |
|---|---|---|---|---|
| ball | 40 | 618..683 | 0.948..0.996 | 0.9981..0.9999 |
| cube | 40 | 624..694 | 0.749..0.867 | 0.9632..0.9851 |

## where the threshold goes

Cubes top out at 0.9851 and balls start at 0.9981, so the two ranges do not touch. Any threshold in (0.9851, 0.9981) separates them; 0.9915 currently sits at 0.9915.
The middle of that gap is 0.9916.

## kind accuracy at the current thresholds

`BALL_FILL` = 0.9915, `BALL_ELONGATION` = 0.907

- correct: 80/80 (100.0%)
