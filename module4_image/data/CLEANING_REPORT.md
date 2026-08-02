# Dataset Cleaning + Merge Report

## What changed from the 3-folder version
Merged `my_dataset_gpay` + `my_dataset_phonepewhite` + `my_dataset_phonepeblack` into ONE
single dataset: `data/real/`, `data/fake/`, `data/meta/`. No more per-app folders.

## Global index ranges (which real_XXXX.jpg came from which app)
| App | Real index range | Count |
|---|---|---|
| gpay | real_0000 – real_0019 | 20 |
| phonepewhite | real_0020 – real_0043 | 24 |
| phonepeblack | real_0044 – real_0122 | 79 |

Every fake image and its meta json were renumbered to match this same global index
(e.g. `fake_0044_0_splice.jpg` is derived from `real_0044.jpg`, which is phonepeblack's
first real image).

## Verification performed after merge
- Manifest (`labels_all.csv`): 492 rows, all paths verified to exist on disk — 0 missing.
- Folder counts match manifest: `real/` = 123, `fake/` = 369, `meta/` = 369.
- **meta → real traceability: 0 mismatches** (previously phonepewhite/phonepeblack had
  broken references — see prior report. The merge rebuilt every `source_image` field from
  the verified filename-index mapping instead of trusting the old broken meta value, so
  this is now fully correct for all 3 original apps, not just gpay.)
- Content-duplicate check (MD5): 0 duplicates after merge — no accidental overwrites.
- Each `meta/*.json` also now has an `origin_app` field (`gpay`/`phonepewhite`/`phonepeblack`)
  so you can still tell which app a sample came from even though it's one merged dataset.

## Folder structure
```
data/
├── real/                  # 123 genuine screenshots, real_0000.jpg .. real_0122.jpg
├── fake/                  # 369 forged screenshots, fake_{real_idx}_{variant}_{type}.jpg
├── meta/                  # 369 bbox + manipulation_type + origin_app jsons, 1 per fake image
├── labels_all.csv         # image_path,label manifest -- use this directly for training
└── generate_forgeries.py  # your original forgery generator (kept for reference)
```

## Adding more real images from here
Since it's one merged pool now, your next real image is **real_0123.jpg** (continue the
global sequence, don't restart per-app). After adding new real images to `data/real/`,
rerun `generate_forgeries.py` against just the new ones, then rebuild `labels_all.csv`:
```powershell
python -c "
import os, csv
rows = []
for cls, label in [('real',0), ('fake',1)]:
    folder = f'data/{cls}'
    for f in sorted(os.listdir(folder)):
        rows.append({'image_path': f'data/{cls}/{f}', 'label': label})
with open('data/labels_all.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image_path','label']); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows written')
"
```
