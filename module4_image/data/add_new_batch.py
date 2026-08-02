"""
add_new_batch.py
-----------------
Adds a folder of NEW real payment screenshots into the existing merged
dataset (data/real/, data/fake/, data/meta/, labels_all.csv), continuing
the global index automatically -- no manual renumbering needed.

What it does, in order:
  1. Looks at data/real/ to find the next free index (e.g. if real_0122.jpg
     is the last one, new photos become real_0123.jpg, real_0124.jpg, ...)
  2. Copies your new real photos in with that numbering.
  3. Generates 3 forged variants per new real photo (chosen from: splice,
     text_overlay, local_inconsistency, recompression, local_recompression
     -- same manipulations as generate_forgeries.py, see that file for
     details), using the SAME logic as generate_forgeries.py, and saves them
     directly into data/fake/ + data/meta/ with matching global indices.
  4. Rebuilds data/labels_all.csv from scratch by scanning the folders
     (always accurate, never drifts out of sync).

Usage (run from inside the `data/` folder):
    python add_new_batch.py --input /path/to/new_real_photos --per_image 3

Your new real photos can have any filenames/extensions (jpg/jpeg/png/webp) --
they'll be renamed to real_XXXX.jpg automatically.
"""

import argparse
import csv
import os
import random
from pathlib import Path

from PIL import Image

from generate_forgeries import MANIPULATIONS


def next_free_index(real_dir: Path) -> int:
    existing = sorted(real_dir.glob("real_*.jpg"))
    if not existing:
        return 0
    last = existing[-1].stem  # "real_0122"
    return int(last.split("_")[1]) + 1


def main(args):
    data_dir = Path(".")
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    meta_dir = data_dir / "meta"
    for d in (real_dir, fake_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    new_images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
    if not new_images:
        raise SystemExit(f"No images found in {input_dir}")

    start_idx = next_free_index(real_dir)
    print(f"Adding {len(new_images)} new real photos, starting at real_{start_idx:04d}.jpg")

    random.seed(args.seed)
    import json as jsonlib

    added_real, added_fake = 0, 0
    for offset, path in enumerate(new_images):
        idx = start_idx + offset
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {path.name}: could not open ({e})")
            continue

        real_name = f"real_{idx:04d}.jpg"
        img.save(real_dir / real_name, quality=95)
        added_real += 1

        chosen_ops = random.sample(MANIPULATIONS, k=min(args.per_image, len(MANIPULATIONS)))
        for j, op in enumerate(chosen_ops):
            try:
                tampered, mtype, bbox = op(img)
            except Exception as e:
                print(f"  [skip manipulation] {op.__name__} on {path.name}: {e}")
                continue
            fake_name = f"fake_{idx:04d}_{j}_{mtype}.jpg"
            # Fixed quality=95, matching real -- see generate_forgeries.py
            # for why: a random quality here let forensic features detect
            # "was this file saved at lower quality" instead of genuine
            # tampering, which is a shortcut that won't generalize.
            tampered.save(fake_dir / fake_name, quality=95)
            added_fake += 1

            with open(meta_dir / f"fake_{idx:04d}_{j}_{mtype}.json", "w") as f:
                jsonlib.dump(
                    {
                        "source_image": real_name,
                        "manipulation_type": mtype,
                        "bbox_xyxy": list(bbox),
                        "origin_app": args.origin_app or "unspecified",
                    },
                    f,
                    indent=2,
                )

    print(f"Added {added_real} real + {added_fake} fake images.")

    # Rebuild labels_all.csv from scratch by scanning folders (always accurate).
    rows = []
    for cls, label in [("real", 0), ("fake", 1)]:
        folder = data_dir / cls
        for f in sorted(os.listdir(folder)):
            rows.append({"image_path": f"data/{cls}/{f}", "label": label})

    with open(data_dir / "labels_all.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    genuine = sum(1 for r in rows if r["label"] == 0)
    forged = sum(1 for r in rows if r["label"] == 1)
    print(f"labels_all.csv rebuilt: {len(rows)} rows ({genuine} genuine / {forged} forged)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder of new real screenshots to add")
    parser.add_argument("--per_image", type=int, default=3, help="Fake variants per new real image (max 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--origin_app", default=None, help="e.g. gpay / phonepewhite / phonepeblack / paytm (optional, just for your own tracking)")
    main(parser.parse_args())
