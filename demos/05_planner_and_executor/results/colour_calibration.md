# colour calibration, top camera

One 5 cm object per colour, all in one rendered frame, and the
pixels each object's known footprint covers were read in HSV. No mask, band or
detector is involved, so this measures the bands the rest of perception depends
on rather than assuming them.

| colour | px | H peak | share | S 5/95 at peak | V 5/95 at peak | current band | nominal H |
|---|---|---|---|---|---|---|---|
| red | 845 | 1 | 100% | 169/187 | 209/219 | 0-11 | 0 |
| green | 857 | 69 | 100% | 153/164 | 190/204 | 39-90 | 64 |
| blue | 856 | 110 | 100% | 174/175 | 212/214 | 104-135 | 116 |
| yellow | 954 | 29 | 100% | 167/180 | 211/220 | 22-38 | 26 |
| white | 967 | 0 | 63% | 0/1 | 216/223 | n/a | 0 |
| black | 964 | 0 | 89% | 0/4 | 68/71 | n/a | 0 |
| orange | 1080 | 24 | 97% | 169/182 | 217/222 | 20-28 | 14 |
| purple | 1078 | 153 | 95% | 94/130 | 174/214 | 144-163 | 156 |
| pink | 1076 | 170 | 79% | 18/201 | 152/224 | 163-177 | 173 |

## how far apart the colours are

| pair | hue gap | S floors | verdict |
|---|---|---|---|
| red / green | 68 | 169-187 / 153-164 | separated by hue |
| red / blue | 71 | 169-187 / 174-175 | separated by hue |
| red / yellow | 28 | 169-187 / 167-180 | separated by hue |
| red / white | 1 | 169-187 / 0-1 | no hue of its own; separable by S/V (S 169-187, V below) |
| red / black | 1 | 169-187 / 0-4 | no hue of its own; separable by S/V (S 169-187, V below) |
| red / orange | 23 | 169-187 / 169-182 | separated by hue |
| red / purple | 28 | 169-187 / 94-130 | separated by hue |
| red / pink | 11 | 169-187 / 18-201 | hue gap 11, saturation overlaps: too close |
| green / blue | 41 | 153-164 / 174-175 | separated by hue |
| green / yellow | 40 | 153-164 / 167-180 | separated by hue |
| green / white | 69 | 153-164 / 0-1 | no hue of its own; separable by S/V (S 153-164, V below) |
| green / black | 69 | 153-164 / 0-4 | no hue of its own; separable by S/V (S 153-164, V below) |
| green / orange | 45 | 153-164 / 169-182 | separated by hue |
| green / purple | 84 | 153-164 / 94-130 | separated by hue |
| green / pink | 79 | 153-164 / 18-201 | separated by hue |
| blue / yellow | 81 | 174-175 / 167-180 | separated by hue |
| blue / white | 70 | 174-175 / 0-1 | no hue of its own; separable by S/V (S 174-175, V below) |
| blue / black | 70 | 174-175 / 0-4 | no hue of its own; separable by S/V (S 174-175, V below) |
| blue / orange | 86 | 174-175 / 169-182 | separated by hue |
| blue / purple | 43 | 174-175 / 94-130 | separated by hue |
| blue / pink | 60 | 174-175 / 18-201 | separated by hue |
| yellow / white | 29 | 167-180 / 0-1 | no hue of its own; separable by S/V (S 167-180, V below) |
| yellow / black | 29 | 167-180 / 0-4 | no hue of its own; separable by S/V (S 167-180, V below) |
| yellow / orange | 5 | 167-180 / 169-182 | hue gap 5: TOO CLOSE, needs a different albedo |
| yellow / purple | 56 | 167-180 / 94-130 | separated by hue |
| yellow / pink | 39 | 167-180 / 18-201 | separated by hue |
| white / black | 0 | 0-1 / 0-4 | no hue of its own; separable by S/V (S 0-1, V below) |
| white / orange | 24 | 0-1 / 169-182 | no hue of its own; separable by S/V (S 0-1, V below) |
| white / purple | 27 | 0-1 / 94-130 | no hue of its own; separable by S/V (S 0-1, V below) |
| white / pink | 10 | 0-1 / 18-201 | no hue of its own; separable by S/V (S 0-1, V below) |
| black / orange | 24 | 0-4 / 169-182 | no hue of its own; separable by S/V (S 0-4, V below) |
| black / purple | 27 | 0-4 / 94-130 | no hue of its own; separable by S/V (S 0-4, V below) |
| black / pink | 10 | 0-4 / 18-201 | no hue of its own; separable by S/V (S 0-4, V below) |
| orange / purple | 51 | 169-182 / 94-130 | separated by hue |
| orange / pink | 34 | 169-182 / 18-201 | separated by hue |
| purple / pink | 17 | 94-130 / 18-201 | hue gap 17, saturation overlaps: too close |
