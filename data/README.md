# Dataset (NOT committed)

No images or labels are stored in git. This folder holds the subset produced by
`phase_b/data_prep.py prepare`, which you must run once in Colab (it needs
~2.7 GB of download and a GPU-free conversion step).

## What is used

**TACO (Trash Annotations in Context)** — CC BY 4.0, verified 2026-11-24 from
[Zenodo record 3587843](https://zenodo.org/records/3587843) and
[tacodataset.org](http://tacodataset.org/). It is litter **in the wild**, not
conveyor footage. That substitution is a stated limitation, not a result.

**ZeroWaste was NOT used.** Its Roboflow page is reachable but contradicts
itself on licence (header "CC BY 4.0", About "CC BY-NC 4.0"), and downloading
the export needs `ROBOFLOW_API_KEY`. See `docs/data_sources.md` and the entry
in `results/failures.md`. Do not substitute it without resolving both.

## Steps (Colab, in order)

```bash
# 1. Download TACO.zip (2.7 GB) from Zenodo, then unzip into data/raw/taco/
python phase_b/data_prep.py download --name taco --dest data/raw

# 2. Convert to a 6-class YOLO subset, scene-disjoint, 1500 train / 300 val.
#    Looks for data/raw/taco/data/annotations.json and the matching images dir.
python phase_b/data_prep.py prepare \
  --json  data/raw/taco/data/annotations.json \
  --images data/raw/taco/data \
  --out   data/waste \
  --train 1500 --val 300 --seed 0

# 3. Verify the inventory (appends a row to results/dataset.csv)
python phase_b/data_prep.py stats --out data/waste
```

`data/waste.yaml` is committed as the expected layout. `prepare` rewrites it
with the real counts from the subset. Class order is `can, bottle,
bottle_cap, cup, plastic_bag_wrapper, other`; the mapping is derived from
TACO's own `detector/taco_config/map_10.csv` and recorded in
`docs/data_sources.md`.

If Zenodo is unreachable, you must download `TACO.zip` by hand from
https://zenodo.org/records/3587843 and place the extracted `annotations.json`
plus image files under `data/raw/taco/data/`. Record the failure in
`results/failures.md` first.
