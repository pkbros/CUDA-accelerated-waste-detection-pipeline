# Data sources (checked 2026-11-24)

This repo will be published, so a licence is stated **only** when it was read
from a source we can link. "Unverified" means exactly that: check it yourself
before publishing or redistributing the images.

Legend: **VERIFIED** = licence string read from the linked source. **UNVERIFIED** =
no licence statement found for the thing we would actually download — check
before publishing.

| Dataset | What it contains | Label type | Size (source-reported) | Licence | Verified? | URL | Download method | Detection fine-tune? |
|---|---|---|---|---|---|---|---|---|
| ZeroWaste (Roboflow `zerowaste-contamination`) | Waste/contamination on conveyor-style imagery; closest to the motivating setting | Roboflow export — request `yolov8` to get images + per-image box `.txt` files | **Unknown**: the Roboflow page returned HTTP 403 from this environment, so no count was confirmed | **Unverified — check before publishing.** The *original academic* dataset is CC BY-NC 4.0 (NonCommercial) per the official repo README; that does not automatically cover this hosted copy | **NO** for the hosted copy. Academic original: **YES**, CC BY-NC 4.0 | [Roboflow Universe](https://universe.roboflow.com/zerowaste-data/zerowaste-contamination); academic original: [dbash/zerowaste](https://github.com/dbash/zerowaste) | `ROBOFLOW_API_KEY` must be set; `phase_b/data_prep.py download --name zerowaste` uses the `roboflow` pip package, otherwise it names the missing variable and skips | Yes, in principle — boxes come from the Roboflow export; blocked here by the missing key **and** unverified licence |
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
