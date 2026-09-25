"""
Expected dataset layout on disk:

    datasets/
        chest/
            normal/      *.jpg / *.png / ...
            abnormal/
        bone/
            normal/
            abnormal/

Add another top-level folder (e.g. datasets/knee/{normal,abnormal}/) to
extend the project to a new body part later -- nothing else in this file
needs to change, only settings.SUPPORTED_BODY_PARTS and one training run.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(folder: Path):
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def list_images_by_bodypart(datasets_root: Path) -> tuple[list[Path], list[str]]:
    """
    For training the body-part router: every image under
    datasets/<body_part>/*/*.ext, labeled by <body_part>, regardless of its
    normal/abnormal subfolder.
    """
    paths: list[Path] = []
    labels: list[str] = []
    for body_part_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in body_part_dir.iterdir() if p.is_dir()):
            for img_path in _iter_images(class_dir):
                paths.append(img_path)
                labels.append(body_part_dir.name)
    return paths, labels


def list_images_by_class(datasets_root: Path, body_part: str) -> tuple[list[Path], list[str]]:
    """
    For training a body-part-specific abnormality classifier: images under
    datasets/<body_part>/<normal|abnormal>/*.ext, labeled by that subfolder
    name.
    """
    paths: list[Path] = []
    labels: list[str] = []
    body_part_dir = datasets_root / body_part
    if not body_part_dir.is_dir():
        raise FileNotFoundError(
            f"No dataset folder at {body_part_dir}. Expected datasets/{body_part}/normal/ "
            f"and datasets/{body_part}/abnormal/ with images in each."
        )
    for class_dir in sorted(p for p in body_part_dir.iterdir() if p.is_dir()):
        for img_path in _iter_images(class_dir):
            paths.append(img_path)
            labels.append(class_dir.name)
    return paths, labels
