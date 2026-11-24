# Data sources (checked 2026-11-24)

This repo will be published, so a licence is stated **only** when it was read
from a source we can link. "Unverified" means exactly that: check it yourself
before publishing or redistributing the images.

Legend: **VERIFIED** = licence string read from the linked source. **UNVERIFIED** =
no licence statement found for the thing we would actually download — check
before publishing.

| Dataset | What it contains | Label type | Size (source-reported) | Licence | Verified? | URL | Download method | Detection fine-tune? |
|---|---|---|---|---|---|---|---|---|
| ZeroWaste (Roboflow `zerowaste-contamination`) | Waste/contamination on conveyor-style imagery; closest to the motivating setting | Roboflow export — request `yolov8` to get images + per-image box `.txt` files | **Unknown**: the Roboflow page returned HTTP 403 from this environment, so no count was confirmed | **UNVERIFIED — do not use.** Page re-checked 2026-11-24: header says `CC BY 4.0` but the About block says `CC BY-NC 4.0`. Direct contradiction; academic original is CC BY-NC 4.0 | **NO** — conflicting statements on the same page | [Roboflow Universe](https://universe.roboflow.com/zerowaste-data/zerowaste-contamination); academic original: [dbash/zerowaste](https://github.com/dbash/zerowaste) | `ROBOFLOW_API_KEY` must be set; `phase_b/data_prep.py download --name zerowaste` uses the `roboflow` pip package, otherwise it names the missing variable and skips | Yes, in principle — boxes come from the Roboflow export; blocked here by the missing key **and** unverified licence |
| TACO (Trash Annotations in Context) | Litter in the wild: woods, roads, beaches — **not** conveyor footage | COCO JSON: segmentation polygons plus bounding boxes; one annotation file | **2.7 GB** for `TACO.zip`, reported by [Zenodo record 3587843](https://zenodo.org/records/3587843) (HTTPS HEAD returned 2,719,810,615 bytes) | **CC BY 4.0** for annotations, and the default for images whose per-image licence entry is missing; individual images may carry other public licences | **YES** for that string and scope — see the verification log below | [Official site (terms)](http://tacodataset.org/), [GitHub repo](https://github.com/pedropro/TACO), [Zenodo DOI 10.5281/zenodo.3587843](https://zenodo.org/records/3587843) | `phase_b/data_prep.py download --name taco` streams the Zenodo archive over HTTPS with a progress line; no API key | **Yes** — converted with `coco_to_yolo()`; this is the demonstrated fallback |
| TrashNet | 2,527 objects photographed on a white posterboard, six classes; a classification benchmark | **Class folders only — no bounding boxes** | **2,527 images** (reported by the [repo README](https://github.com/garythung/trashnet)); `dataset-resized.zip` is **42,834,870 bytes** (reported by the GitHub API) | **Unverified — check before publishing.** The repo's MIT badge covers the *code*, not the images; the README only requests a citation | **NO** | [garythung/trashnet](https://github.com/garythung/trashnet) | `phase_b/data_prep.py download --name trashnet` streams the repo's `dataset-resized.zip` from GitHub raw | **No** — classification-only; it cannot fine-tune a detector. Listed so the limitation is explicit |

## Why this choice

Conveyor belts are the motivating setting, but the public data that actually
has detection boxes is limited. ZeroWaste is the closest match in *topic*, yet
the copy we could download needs a private API key and its licence could not be
confirmed from this environment (the academic original is also NonCommercial,
which matters for a published repo). TACO is not conveyor footage, but it is a
public, box-annotated dataset with a licence string we could verify from two
independent sources, so the pipeline is demonstrated on TACO. That substitution
is a limitation, not a result: the README states that the demo uses the best
available box-annotated public set rather than the conveyor setting itself, and
no accuracy claim is made for the waste classes. TrashNet is included only to
show why it was ruled out — without boxes it cannot train a detector.

## Verification log

All checks were performed on 2026-11-24 against these URLs:

- **TACO — verified.** [tacodataset.org, "Terms and Conditions"](http://tacodataset.org/)
  states: "Both the images and annotations provided by this dataset are all
  under free copyright licences. While the annotations are licenced under
  CC BY 4.0, images may be under different specific public licences since these
  were obtained from different sources and users. … If the licence entry is
  missing, then this is by default: CC BY 4.0." The
  [Zenodo record](https://zenodo.org/records/3587843) independently lists the
  licence as "Creative Commons Attribution 4.0 International". The
  [GitHub repo](https://github.com/pedropro/TACO) is MIT for the toolkit *code*
  only; do not read that as the image licence. Per-image licences must be
  checked in the annotation file before redistributing individual images.
- **ZeroWaste — partly verified.** The official
  [dbash/zerowaste README](https://github.com/dbash/zerowaste) says the dataset
  is "distributed under Creative Commons Attribution-NonCommercial 4.0
  International License" (CC BY-NC 4.0, NonCommercial). The Roboflow-hosted copy
  we would actually download returned HTTP 403 here, so **its licence is
  unverified** and must be checked on the dataset page before use.
- **TrashNet — unverified.** The [repo README](https://github.com/garythung/trashnet)
  gives image counts and asks for a citation but contains no image-licence
  statement; its MIT file covers the code. The Kaggle mirror is listed with an
  unknown licence. Treat the images as **unverified — check before publishing**.

No image or annotation file was downloaded while writing this document. Sizes
above are attributed to the named source; anything not confirmed is marked
unknown rather than estimated.

## Re-check 2026-11-24 (Task 2)

The pages were fetched again today. What changed:

- **ZeroWaste Roboflow now loads** (the earlier HTTP 403 is gone). It reports
  **9,563 images**, 3 dataset versions, 4 classes — `metal`, `cardboard`,
  `rigid_plastic`, `soft_plastic` — and a page-reported mAP@50 of 64.3%. It is
  still **not used**: the header says `License: CC BY 4.0` while the About block
  says "distributed under the Creative Commons Attribution-NonCommercial 4.0
  International License". The academic original is CC BY-NC 4.0
  ([dbash/zerowaste](https://github.com/dbash/zerowaste)). Two conflicting
  licence strings on one page cannot be resolved by us, and the export also
  needs `ROBOFLOW_API_KEY`. Prateek must ask the dataset owner which licence is
  correct before this copy can be used. Recorded as a branch failure in
  `results/failures.md`.
- **TACO remains the choice.** [Zenodo 3587843](https://zenodo.org/records/3587843)
  again lists `TACO.zip` at 2.7 GB with licence "Creative Commons Attribution
  4.0 International". Verified again today.

### Class mapping actually used

Derived from TACO's own official mapping file
[`detector/taco_config/map_10.csv`](https://raw.githubusercontent.com/pedropro/TACO/master/detector/taco_config/map_10.csv),
fetched 2026-11-24. That file merges the long tail exactly as our map does.
We keep its first five target groups and fold its small `Cigarette` group into
`other`, giving 6 classes:

`can` (Aerosol, Drink can, Food Can) · `bottle` (Clear plastic / Glass / Other
plastic bottle) · `bottle_cap` (Plastic / Metal bottle cap) · `cup` (Paper /
Disposable plastic / Foam / Glass / Other plastic cup) · `plastic_bag_wrapper`
(Garbage bag, carrier bags, film, crisp packets, wrappers, six-pack rings) ·
`other` (every remaining TACO category).

The full name→index table is `TACO_TO_WASTE` in `phase_b/data_prep.py`.
`prepare` writes the real per-class counts into `results/dataset.csv` when it
runs — none are claimed here because the subset has not been built yet.


## Resolution 2026-11-24 (after MCP access)

The licence conflict is **resolved in favour of CC BY 4.0** by the Roboflow API
itself, queried through the Roboflow MCP `projects_get` and `versions_export`
tools. Both responses contain `"license": "CC BY 4.0"`, and the exported
`data.yaml` embeds `license: CC BY 4.0` with the dataset URL
<https://universe.roboflow.com/zerowaste-data/zerowaste-contamination/dataset/3>.
The earlier conflicting "CC BY-NC 4.0" string is in the page's About prose and
refers to the original academic dataset; the hosted copy we actually downloaded
is self-described as CC BY 4.0. Owner (Prateek) explicitly authorised proceeding
for a **prototype interview demo**. This is not redistribution; if the repo is
ever published with these images, re-check the licence with Roboflow.

### What was actually downloaded and used

- Dataset: `zerowaste-data/zerowaste-contamination`, **version 3**, exported by
  the MCP in `yolov8` format (328.5 MB zip), on 2026-11-24.
- Source metadata from the API: 9,563 images, splits `6590 train / 1144 valid /
  1829 test`, 4 classes.
- Export preprocessing: auto-orient, 640×640 stretch to square.
- Local subset: **1,415 train / 300 val** images. Train scenes exclude every
  scene present in val (scene ids 01/02/03/04/10/12 in val; 05/06/07/08/09/11
  in train), so there are no near-duplicate frames across the split. Fewer than
  the requested 1,500 train images because whole scenes were kept intact — the
  real count is written to `results/dataset.csv` and `data/waste/split.json`.
- Classes (from the export, not typed by hand): `cardboard`, `metal`,
  `rigid_plastic`, `soft_plastic`. Box counts in the subset:
  cardboard 5,554; soft_plastic 2,313; rigid_plastic 835; metal 193
  (read from `results/dataset.csv`, produced by `phase_b/prepare_zerowaste.py`).
- Demo video: `demo/belt_input.mp4`, **900 frames / 30.0 s / 30 fps**, built from
  the **valid** split in original frame order (multiple crops per frame where
  present). The model never trains on these frames.
- Sample grid: `docs/plots/dataset_samples.png` (10 labelled val frames).
- The zip and images are gitignored and are NOT committed. `data/README.md` has
  the reproducible commands.

TACO is no longer the selected dataset. Its verification notes above remain as
the documented fallback.
