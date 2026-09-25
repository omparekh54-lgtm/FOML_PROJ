"""
Assemble the training dataset from the raw Kaggle folders.

Usage (from the project root):
    python -m ml.training.build_dataset --source <folder containing chest_xray/ and Bone_Fracture_Binary_Classification/>

Produces:
    datasets/chest/{normal,abnormal}/   <- sampled from chest_xray/train/{NORMAL,PNEUMONIA}
    datasets/bone/{normal,abnormal}/    <- from Bone_Fracture_Binary_Classification/.../{train,val,test}/{not fractured,fractured}
    analyzer/static/analyzer/samples/   <- a few held-out images shown as "try a sample" in the UI

Rules applied, because the raw folders are messy:
  * The bone dataset is heavily augmented ("12-rotated1-rotated3.jpg",
    "x - Copy.jpg", "x (1).jpg"); those near-duplicates are skipped so the
    model isn't graded on rotated copies of its own training images.
  * The same picture often appears in several splits under different names,
    so files are de-duplicated by content hash.
  * Files Pillow cannot fully decode (a few are truncated) are skipped.
  * Sampling is seeded, so every build produces the same dataset.
  * Sample images are removed from the pool *before* training, so the demo
    images in the UI are ones the models have never seen.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUGMENTED = re.compile(r"(-rotated\d)|( - Copy)|(\(\d+\))", re.IGNORECASE)
BONE_ROOT = Path("Bone_Fracture_Binary_Classification") / "Bone_Fracture_Binary_Classification"


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _clean_pool(paths: list[Path], seen_hashes: set[str]) -> list[Path]:
    """Drop augmented copies, duplicates (by content) and unreadable files."""
    pool = []
    for p in paths:
        if AUGMENTED.search(p.stem):
            continue
        h = _digest(p)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        pool.append(p)
    return pool


def _take(pool: list[Path], n: int, rng: random.Random) -> list[Path]:
    """Seeded sample of up to n readable files from pool."""
    shuffled = pool[:]
    rng.shuffle(shuffled)
    picked = []
    for p in shuffled:
        if len(picked) >= n:
            break
        if _readable(p):
            picked.append(p)
    return picked


def _copy(files: list[Path], dest: Path, prefix_with_split: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = f"{f.parent.parent.name}_{f.name}" if prefix_with_split else f.name
        shutil.copy2(f, dest / name.replace(" ", "_"))


def _save_sample(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("L")
        img.thumbnail((640, 640))
        img.save(dest, format="JPEG", quality=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(PROJECT_ROOT))
    parser.add_argument("--dest", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--samples-dir", default=str(PROJECT_ROOT / "analyzer" / "static" / "analyzer" / "samples"))
    parser.add_argument("--chest-per-class", type=int, default=450)
    parser.add_argument("--bone-per-class", type=int, default=450)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source, dest, samples_dir = Path(args.source), Path(args.dest), Path(args.samples_dir)
    rng = random.Random(args.seed)

    if dest.exists():
        shutil.rmtree(dest)
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # ---- chest -------------------------------------------------------------
    # Training images come from train/; the tiny val/ split is never trained
    # on, so it is the natural source of demo samples.
    chest_sources = {
        "normal": ("NORMAL", source / "chest_xray" / "train" / "NORMAL"),
        "abnormal": ("PNEUMONIA", source / "chest_xray" / "train" / "PNEUMONIA"),
    }
    for label, (raw_name, folder) in chest_sources.items():
        pool = _clean_pool(_images(folder), set())
        picked = _take(pool, args.chest_per_class, rng)
        _copy(picked, dest / "chest" / label, prefix_with_split=False)
        print(f"[dataset] chest/{label}: {len(picked)} images (pool {len(pool)})")

        val_pool = _images(source / "chest_xray" / "val" / raw_name)
        for i, s in enumerate(_take(val_pool, args.samples_per_class, rng)):
            _save_sample(s, samples_dir / f"chest_{label}_{i + 1}.jpg")

    # ---- bone --------------------------------------------------------------
    bone_pools: dict[str, list[Path]] = {}
    bone_hashes: dict[str, set[str]] = {}
    for label, raw_name in (("normal", "not fractured"), ("abnormal", "fractured")):
        raw = []
        for split in ("train", "val", "test"):
            raw += _images(source / BONE_ROOT / split / raw_name)
        bone_hashes[label] = set()
        bone_pools[label] = _clean_pool(raw, bone_hashes[label])
    # A picture filed under both "fractured" and "not fractured" is label
    # noise; drop it from both rather than guess.
    conflicting = bone_hashes["normal"] & bone_hashes["abnormal"]
    if conflicting:
        print(f"[dataset] bone: dropping {len(conflicting)} image(s) labelled both ways")

    for label in ("normal", "abnormal"):
        pool = [p for p in bone_pools[label] if _digest(p) not in conflicting]
        # Hold out demo samples first so they are never trained on.
        samples = _take(pool, args.samples_per_class, rng)
        pool = [p for p in pool if p not in samples]
        for i, s in enumerate(samples):
            _save_sample(s, samples_dir / f"bone_{label}_{i + 1}.jpg")
        picked = _take(pool, args.bone_per_class, rng)
        _copy(picked, dest / "bone" / label, prefix_with_split=True)
        print(f"[dataset] bone/{label}: {len(picked)} images (pool {len(pool)} after "
              f"removing augmented copies and duplicates)")

    print(f"[dataset] samples -> {samples_dir} ({len(list(samples_dir.glob('*.jpg')))} files)")


if __name__ == "__main__":
    main()
