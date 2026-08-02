"""
generate_forgeries.py

Purpose
-------
Takes a folder of REAL transaction screenshots (yours/your friends', with
consent) and produces a labeled dataset of controlled, synthetic tampering
for training a fake-screenshot / image-forgery DETECTOR.

This does NOT try to imitate any specific payment app's UI. It applies
generic image-forensics perturbations (splicing, recompression, text
overlay, local color/noise inconsistency) that are the standard way
forgery-detection datasets (CASIA, IMD2020, etc.) are built.

Output structure
-----------------
output/
  real/                  <- untouched copies of your originals (label 0)
  fake/                  <- tampered versions (label 1)
  labels.csv             <- filename,label,manipulation_type,region_bbox
  meta/<name>.json        <- per-image manipulation metadata (for localization tasks)

Usage
-----
python generate_forgeries.py --input /path/to/real_screenshots --output /path/to/output --per_image 3
"""

import os
import io
import json
import random
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


# ----------------------------- helpers ----------------------------- #

def load_image(path):
    return Image.open(path).convert("RGB")


def feathered_paste(base: Image.Image, patch: Image.Image, box, feather_px: int = 6) -> Image.Image:
    """Paste `patch` into `base` at `box` with a soft-edged alpha mask
    instead of a hard rectangle. A hard paste boundary is one of the
    easiest things for both a human eye and a naive detector to spot --
    real edits (careful Photoshop work, or the fake-screenshot generator
    apps actual fraudsters use) essentially never leave a visible hard
    edge. Feathering the boundary removes that easy tell, forcing the
    detector to rely on genuine pixel-statistics evidence (ELA, noise,
    compression mismatch) instead of "is there a suspicious rectangle".
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return base

    mask = Image.new("L", (w, h), 255)
    # shrink the fully-opaque core so there's room for the blurred border
    # to actually fall off to 0 before hitting the patch edge
    inner = ImageDraw.Draw(mask)
    inset = min(feather_px, w // 2, h // 2)
    if inset > 0:
        inner.rectangle([inset, inset, w - inset, h - inset], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px / 2))

    out = base.copy()
    out.paste(patch, (x0, y0), mask)
    return out


def poisson_blend(base: Image.Image, patch: Image.Image, box) -> Image.Image:
    """Seamless (Poisson) blend of `patch` into `base` at `box`, using
    OpenCV's seamlessClone. This is the standard technique real forgery
    datasets (and real photo editors) use for splice compositing --
    it matches local gradients/lighting at the boundary instead of just
    copying raw pixels, so the splice doesn't leave a flat color-mismatch
    edge the way a plain paste does. Falls back to feathered_paste if
    seamlessClone rejects the geometry (can happen with very thin/edge-
    touching regions).
    """
    x0, y0, x1, y1 = box
    base_bgr = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
    patch_bgr = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    mask = 255 * np.ones(patch_bgr.shape[:2], dtype=np.uint8)
    center = (x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2)
    try:
        blended = cv2.seamlessClone(patch_bgr, base_bgr, mask, center, cv2.NORMAL_CLONE)
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    except cv2.error:
        return feathered_paste(base, patch, box)


def random_font(size):
    """Try a few common system fonts, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def random_region(w, h, min_frac=0.08, max_frac=0.25):
    """Pick a random rectangular region (good proxy for an amount/name field)."""
    rw = int(w * random.uniform(min_frac, max_frac))
    rh = int(h * random.uniform(0.03, 0.08))
    x0 = random.randint(0, max(1, w - rw - 1))
    y0 = random.randint(0, max(1, h - rh - 1))
    return (x0, y0, x0 + rw, y0 + rh)


# ------------------------- manipulation ops ------------------------- #

def op_splice(img):
    """Copy a patch from elsewhere in the image and blend it over a region,
    simulating a copy-move / splice edit (classic forgery type). Uses
    Poisson (seamlessClone) blending so the boundary matches local
    lighting/gradients instead of leaving a flat, easily-spotted paste
    edge -- this is how real forgery-detection benchmark datasets (CASIA,
    IMD2020) and real photo-editing tools construct splices."""
    w, h = img.size
    dst = random_region(w, h)
    dw, dh = dst[2] - dst[0], dst[3] - dst[1]

    # source patch: same size, different location
    for _ in range(10):
        sx0 = random.randint(0, max(1, w - dw - 1))
        sy0 = random.randint(0, max(1, h - dh - 1))
        src = (sx0, sy0, sx0 + dw, sy0 + dh)
        # avoid overlapping source/dest
        if not (abs(src[0] - dst[0]) < dw and abs(src[1] - dst[1]) < dh):
            break

    patch = img.crop(src)
    # slight recompression / smoothing on the patch so it blends imperfectly
    patch = patch.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.6)))
    out = poisson_blend(img, patch, dst)
    return out, "splice", dst


def op_text_overlay(img):
    """Overwrite a region with a solid patch + new text, simulating a
    changed amount / name / status field with font-rendering mismatch.
    Text is rendered anti-aliased (2x supersample + downscale, matching
    how real UI text and most photo editors render text) and blended in
    with feathered edges instead of a flat hard-edged rectangle."""
    w, h = img.size
    region = random_region(w, h)
    rw, rh = region[2] - region[0], region[3] - region[1]

    # sample background color near region to (imperfectly) blend
    bg_sample = img.crop(region).resize((1, 1)).getpixel((0, 0))

    # render at 2x then downscale for anti-aliased text edges
    scale = 2
    patch = Image.new("RGB", (rw * scale, rh * scale), bg_sample)
    draw = ImageDraw.Draw(patch)

    fake_texts = ["9,999", "1,50,000", "SUCCESS", "Rahul K.", "10,000", "COMPLETED"]
    text = random.choice(fake_texts)
    font_size = max(10, (rh - 2)) * scale
    font = random_font(font_size)
    text_color = (
        max(0, bg_sample[0] - random.randint(80, 150)),
        max(0, bg_sample[1] - random.randint(80, 150)),
        max(0, bg_sample[2] - random.randint(80, 150)),
    )
    draw.text((2 * scale, 0), text, fill=text_color, font=font)
    patch = patch.resize((rw, rh), Image.LANCZOS)

    # add a touch of matching grain so the patch doesn't look unnaturally
    # smooth/clean next to the original screenshot's natural JPEG noise
    arr = np.array(patch).astype(np.int16)
    grain = np.random.normal(0, random.uniform(1, 3), arr.shape)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    patch = Image.fromarray(arr)

    out = feathered_paste(img, patch, region, feather_px=3)
    return out, "text_overlay", region


def op_local_inconsistency(img):
    """Locally perturb brightness/contrast/noise in a region — mimics the
    subtle lighting/shadow mismatch left behind by editing software."""
    w, h = img.size
    region = random_region(w, h, 0.15, 0.35)
    patch = img.crop(region)

    patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.75, 1.3))
    patch = ImageEnhance.Contrast(patch).enhance(random.uniform(0.75, 1.3))

    arr = np.array(patch).astype(np.int16)
    noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    patch = Image.fromarray(arr)

    out = feathered_paste(img, patch, region, feather_px=8)
    return out, "local_inconsistency", region


def op_recompression(img):
    """Multiple JPEG re-encode cycles at varying quality — creates
    compression-artifact mismatches typical of edited-then-resaved images.
    Applied globally, so bbox is the full image."""
    out = img
    cycles = random.randint(2, 4)
    for _ in range(cycles):
        buf = io.BytesIO()
        q = random.randint(35, 80)
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    w, h = img.size
    return out, "recompression", (0, 0, w, h)


def op_local_recompression(img):
    """Recompress ONE region at a different JPEG quality than the rest of
    the image, then blend it back in -- no visible content change at all.
    This mirrors a very common real fraud technique: crop a field (e.g. the
    amount) from a *different* genuine screenshot and composite it in, or
    export just that region through a separate editing step. The visual
    content can look completely legitimate; the only forensic trace is a
    local compression-history mismatch, which is exactly what ELA-style
    features are meant to catch. This is a meaningfully different (and
    harder) case than op_recompression, which changes compression globally
    and uniformly."""
    w, h = img.size
    region = random_region(w, h, 0.10, 0.30)
    patch = img.crop(region)

    buf = io.BytesIO()
    q = random.randint(40, 75)  # noticeably different from the base quality=95
    patch.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    patch = Image.open(buf).convert("RGB")

    out = feathered_paste(img, patch, region, feather_px=4)
    return out, "local_recompression", region


MANIPULATIONS = [op_splice, op_text_overlay, op_local_inconsistency, op_recompression, op_local_recompression]


# ------------------------------ main ------------------------------ #

def process(input_dir, output_dir, per_image, seed):
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    meta_dir = output_dir / "meta"
    for d in (real_dir, fake_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    rows = []
    for idx, path in enumerate(sorted(images)):
        try:
            img = load_image(path)
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue

        # 1. save untouched real copy
        real_name = f"real_{idx:04d}{path.suffix.lower()}"
        img.save(real_dir / real_name, quality=95)
        rows.append([real_name, 0, "none", ""])

        # 2. generate N tampered variants per image
        chosen_ops = random.sample(MANIPULATIONS, k=min(per_image, len(MANIPULATIONS)))
        for j, op in enumerate(chosen_ops):
            try:
                tampered, mtype, bbox = op(img)
            except Exception as e:
                print(f"  manipulation {op.__name__} failed on {path.name}: {e}")
                continue
            fake_name = f"fake_{idx:04d}_{j}_{mtype}{path.suffix.lower()}"
            # IMPORTANT: save at the SAME fixed quality as the real copy above
            # (quality=95), not a random quality. Randomizing this made every
            # fake image's *global* JPEG save-quality differ systematically
            # from every real image's, regardless of manipulation type --
            # forensic features like ELA (vision/ela_features.py) then just
            # learn "was this file saved at a lower quality" instead of
            # detecting genuine local tampering, which won't generalize to
            # real fraud screenshots (those aren't guaranteed to be
            # lower-quality than genuine ones). op_recompression's own
            # internal multi-cycle re-encode (lines above, quality 35-80)
            # is unaffected by this and remains a legitimate signal for that
            # specific manipulation type.
            tampered.save(fake_dir / fake_name, quality=95)
            rows.append([fake_name, 1, mtype, list(bbox)])

            with open(meta_dir / f"{Path(fake_name).stem}.json", "w") as f:
                json.dump(
                    {
                        "source_image": path.name,
                        "manipulation_type": mtype,
                        "bbox_xyxy": list(bbox),
                    },
                    f,
                    indent=2,
                )

    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "manipulation_type", "bbox_xyxy"])
        writer.writerows(rows)

    n_real = sum(1 for r in rows if r[1] == 0)
    n_fake = sum(1 for r in rows if r[1] == 1)
    print(f"Done. {n_real} real, {n_fake} fake images written to {output_dir}")
    print(f"Labels: {output_dir / 'labels.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder of your real screenshots")
    parser.add_argument("--output", required=True, help="Where to write the dataset")
    parser.add_argument("--per_image", type=int, default=3, help="How many fake variants per real image (max 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process(args.input, args.output, args.per_image, args.seed)
