"""
Stage 1: train the body-part router.

Usage (from the project root, /home/claude/xray_analyzer):
    python -m ml.training.train_body_part_router

Reads every image under datasets/<body_part>/{normal,abnormal}/, labels
each one by its top-level body-part folder, reduces them with PCA, and
fits a multi-class classifier on top -- so at inference time an uploaded
image gets projected into the same PCA space and routed to whichever
body-part-specific model (see train_abnormality_classifier.py) applies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.preprocessing import batch_preprocess
from ml.training.dataset_utils import list_images_by_bodypart

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "ml" / "trained_models"))
    parser.add_argument("--n-components", type=int, default=60,
                         help="PCA components for the router. Fewer classes than the "
                              "per-body-part models need, so this can be smaller.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[router] scanning {datasets_root} ...")
    paths, labels = list_images_by_bodypart(datasets_root)
    if not paths:
        raise SystemExit(
            f"No images found under {datasets_root}. Expected "
            f"datasets/<body_part>/<normal|abnormal>/*.jpg "
            f"(see ml/training/dataset_utils.py docstring for the exact layout)."
        )
    print(f"[router] found {len(paths)} images across body parts: "
          f"{sorted(set(labels))}")

    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=args.test_size, stratify=labels, random_state=42,
    )

    print(f"[router] preprocessing {len(train_paths)} training images ...")
    X_train, train_valid = batch_preprocess(train_paths)
    y_train = [y_train[i] for i in train_valid]
    print(f"[router] preprocessing {len(test_paths)} test images ...")
    X_test, test_valid = batch_preprocess(test_paths)
    y_test = [y_test[i] for i in test_valid]

    n_components = min(args.n_components, X_train.shape[0] - 1, X_train.shape[1])
    pipeline = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=42)),
        # Standardising the components before an RBF-kernel SVM beat plain
        # logistic regression in cross-validation (fewer chest/bone mix-ups,
        # notably on hand X-rays, whose bright-bone-on-dark look can resemble
        # a chest film along the first few components).
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])

    print(f"[router] fitting PCA({n_components}) + standardise + RBF SVM ...")
    pipeline.fit(X_train, y_train)

    explained = pipeline.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"[router] PCA retains {explained:.1%} of variance with {n_components} components")

    y_pred = pipeline.predict(X_test)
    print("[router] test-set performance:")
    print(classification_report(y_test, y_pred))

    out_path = models_dir / "body_part_router.joblib"
    joblib.dump(pipeline, out_path)
    print(f"[router] saved -> {out_path}")

    # Out-of-distribution statistics. PCA learned a low-dimensional subspace
    # that real training X-rays live close to. For any image we can ask what
    # fraction of its own pixel variation those components explain
    # (1 - reconstruction error / image variance). Real X-rays score high;
    # noise, text, or unrelated pictures can't be rebuilt from "X-ray
    # components" and score near zero or below. Raw reconstruction error
    # alone doesn't work: a blank image is trivially easy to reconstruct.
    pca = pipeline.named_steps["pca"]
    X_all = np.vstack([X_train, X_test])
    recon = pca.inverse_transform(pca.transform(X_all))
    errors = np.mean((X_all - recon) ** 2, axis=1)
    variances = X_all.var(axis=1)
    explained = 1.0 - errors / np.maximum(variances, 1e-9)
    ood_stats = {
        "explained_min": float(explained.min()),
        "explained_p1": float(np.percentile(explained, 1)),
        "explained_median": float(np.median(explained)),
        "variance_min": float(variances.min()),
    }
    joblib.dump(ood_stats, models_dir / "router_ood_stats.joblib")
    joblib.dump({
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "n_components": int(n_components),
        "variance_retained": float(pca.explained_variance_ratio_.sum()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }, models_dir / "router_metrics.joblib")
    print(f"[router] reconstruction-error stats (for out-of-distribution check): {ood_stats}")


if __name__ == "__main__":
    main()
