# Demo images for notebooks 04a and 06

Keep `USE_DEMO_IMAGES=True` and `IMAGE_PATH=''` to download all five originals
when you run either notebook in Colab, unless the host is unavailable/rate-limiting.
No image upload or annotations are needed when at least one demo photo is available.
This is an unlabelled visual smoke test, not a training set or an accuracy benchmark.
The originals are unmodified; generated overlays are separate files. Preserve
attribution and applicable share-alike terms when redistributing images/overlays.

| Downloaded file | Source and author | License | Scene |
|---|---|---|---|
| bolts_nuts_washers.jpg | [Bolts,Nuts & Washers — Sbrennan83](https://commons.wikimedia.org/wiki/File:Bolts,Nuts_%26_Washers.jpg) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Original fastener photo |
| angle_bracket.jpg | [Bronze angle bracket — Fructibus](https://commons.wikimedia.org/wiki/File:Bronze_angle_bracket_-_4_x_4_cm_-_A.jpg) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Single formed bracket |
| four_gray_brackets.jpg | [4 x Gray angle bracket — Fructibus](https://commons.wikimedia.org/wiki/File:4_x_Gray_angle_bracket_-_12.5_cm_-_A.jpg) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Multiple formed brackets |
| amplifier_bracket.jpg | [Bracket for amp. — Robbie Sproule](https://commons.wikimedia.org/wiki/File:Bracket_for_amp._(25960984807).jpg) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | Stereo pair: repeated views of one bracket, not independent objects |
| channel_connector.jpg | [Hilti Schienen Eckverbinder 90 degrees — Best Tech Nick 25](https://commons.wikimedia.org/wiki/File:Hilti_Schienen_Eckverbinder_90%C2%B0.jpg) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Assembled bracket/channel: clutter test, not loose parts |

The downloader checks each original against its published SHA-1 checksum, decodes
it before saving, and preserves existing files. Per-file sources, licenses and
notes are saved in `ROOT/demo_images/sources.json` and `ATTRIBUTION.txt`. This shared
Drive cache avoids fetching the same photos in every new run. Requests are paced,
transient errors get bounded retries, and numeric/date Retry-After values are
respected. A server delay above 60 seconds is deferred rather than retried early.
Persistent HTTP 429 stops further downloads in that call; cached photos are still
checked and usable. `download_report.json` records available and skipped photos.
The notebook warns and continues with a nonempty verified subset, or raises with
upload instructions if none are available. Checksums are never bypassed.

For an older notebook that failed after downloading some photos, set
`USE_DEMO_IMAGES=False` and `IMAGE_PATH=str(RUN / 'demo_images')` in the current
session, then rerun the photo-loading cell. This uses those images without any
new image downloads. In a new session, use that older folder's absolute Drive path.

## Closer matches to the supplied stamped-part reference

The reference shows irregular bent/stamped sheet-metal parts with holes and tabs,
not just bolts or standard L-brackets. The following manufacturer pages show
closer examples, but no clear open reuse license was verified. They are browsing
references only: their photographs are **not** automatically downloaded, bundled,
or presented as an open training dataset.

- [Sheet metal components — GHS Fasteners](https://www.nutboltswashers.com/sheet-metal-components/): assorted clips, plates and formed parts.
- [Sheet metal/stamping parts — Botrust](https://botrusttech.com/parts/stamping/): clips, catches and chassis components.

To test the image you supplied, upload it to a new Drive photo folder, then set:

```python
USE_DEMO_IMAGES = False
IMAGE_PATH = '/content/drive/MyDrive/Robotekin2026/stamped_part_photos'
```

You can also leave IMAGE_PATH empty with demos disabled and upload your image
when prompted. The clipboard path on your Windows laptop is not accessible in
Colab. Use images you own or have permission to use for a larger collection.

For notebook 04a, try `PROMPTS['bracket'] = 'sheet metal bracket'` or
`'stamped metal part'` in separate runs. Both map to the existing `bracket` output
category; no new planner classes are introduced. These prompts are suggestions,
not validated improvements. Changing prompts requires preparing a new bundle;
notebook 06 evaluates the prompts already embedded in the chosen bundle.

Adding images does not retrain YOLOE or guarantee improved masks. Unusual stamped
parts may need visual exemplars or custom instance-mask fine-tuning. The current
notebook supports text prompts only, and no predictions on these new photos have
been run by the coding agent.
