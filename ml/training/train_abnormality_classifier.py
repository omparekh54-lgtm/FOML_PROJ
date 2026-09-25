"""
Stage 2: train a normal-vs-abnormal model for ONE body part, plus the
surrogate "reasoning" decision tree that explains it.

Usage (from the project root, /home/claude/xray_analyzer):
    python -m ml.training.train_abnormality_classifier --body-part chest
    python -m ml.training.train_abnormality_classifier --body-part bone

Reads datasets/<body_part>/{normal,abnormal}/*.jpg, fits a dedicated PCA
basis for that body part (a chest PCA basis is meaningless for a hand
X-ray, so each body part gets its own), trains a classifier on the PCA
components, then trains a shallow decision tree to mimic that classifier
so its decisions can be read out as plain-language rules (see ml/explain.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from ml.explain import label_components_by_region, train_reasoning_tree
from ml.pipeline import ABNORMALITY_CLASS_NAMES
from ml.preprocessing import batch_preprocess
from ml.training.dataset_utils import list_images_by_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Must match ml.pipeline.ABNORMALITY_CLASS_NAMES = {0: "normal", 1: "abnormal"}
LABEL_TO_INT = {"normal": 0, "abnormal": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-part", required=True, help="e.g. chest, bone")
    parser.add_argument("--datasets-root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "ml" / "trained_models"))
    parser.add_argument("--n-components", default="auto",
                         help="PCA components, or 'auto' to pick by 5-fold cross-validation "
                              "on the training split from --candidates.")
    parser.add_argument("--candidates", default="10,15,20,30,50",
                         help="Component counts tried when --n-components=auto.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tree-depth", default="auto",
                         help="Max depth of the reasoning tree, or 'auto' for the shallowest "
                              "depth (3-6) reaching 85%% fidelity on a validation split. "
                              "Kept small so the explanation stays readable.")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    body_part = args.body_part

    print(f"[{body_part}] scanning {datasets_root / body_part} ...")
    paths, str_labels = list_images_by_class(datasets_root, body_part)
    if not paths:
        raise SystemExit(f"No images found for body part '{body_part}'.")
    unknown = set(str_labels) - set(LABEL_TO_INT)
    if unknown:
        raise SystemExit(
            f"Unexpected class folder name(s) {unknown} under datasets/{body_part}/. "
            f"Expected only 'normal' and 'abnormal'."
        )
    labels = np.array([LABEL_TO_INT[l] for l in str_labels])
    print(f"[{body_part}] found {len(paths)} images "
          f"({(labels == 0).sum()} normal / {(labels == 1).sum()} abnormal)")

    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=args.test_size, stratify=labels, random_state=42,
    )

    print(f"[{body_part}] preprocessing {len(train_paths)} training images ...")
    X_train, train_valid = batch_preprocess(train_paths)
    y_train = y_train[train_valid]
    print(f"[{body_part}] preprocessing {len(test_paths)} test images ...")
    X_test, test_valid = batch_preprocess(test_paths)
    y_test = y_test[test_valid]

    max_components = min(X_train.shape[0] - 1, X_train.shape[1])
    if args.n_components == "auto":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = {}
        for n in (int(c) for c in args.candidates.split(",")):
            n = min(n, max_components)
            model = make_pipeline(
                PCA(n_components=n, random_state=42),
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            )
            scores[n] = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy").mean()
            print(f"[{body_part}]   cv balanced accuracy with {n:>3} components: {scores[n]:.1%}")
        n_components = max(scores, key=scores.get)
        print(f"[{body_part}] picked {n_components} components")
    else:
        n_components = min(int(args.n_components), max_components)

    print(f"[{body_part}] fitting PCA({n_components}) ...")
    pca = PCA(n_components=n_components, random_state=42)
    train_components = pca.fit_transform(X_train)
    test_components = pca.transform(X_test)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[{body_part}] PCA retains {explained:.1%} of variance with {n_components} components")

    print(f"[{body_part}] fitting main classifier (LogisticRegression) ...")
    main_model = LogisticRegression(max_iter=3000, class_weight="balanced")
    main_model.fit(train_components, y_train)

    y_pred = main_model.predict(test_components)
    print(f"[{body_part}] main classifier test-set performance:")
    print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))

    train_main_predictions = main_model.predict(train_components)
    if args.tree_depth == "auto":
        # Choose depth on a validation slice of the TRAINING data so the
        # fidelity reported on the test set below stays an honest estimate.
        fit_c, val_c, fit_p, val_p = train_test_split(
            train_components, train_main_predictions, test_size=0.25,
            stratify=train_main_predictions, random_state=42,
        )
        fidelity_by_depth = {}
        for depth in (3, 4, 5, 6):
            t = train_reasoning_tree(fit_c, fit_p, max_depth=depth)
            fidelity_by_depth[depth] = accuracy_score(val_p, t.predict(val_c))
        good = [d for d, f in fidelity_by_depth.items() if f >= 0.85]
        tree_depth = min(good) if good else max(fidelity_by_depth, key=fidelity_by_depth.get)
        print(f"[{body_part}] validation fidelity by depth: "
              + ", ".join(f"{d}: {f:.1%}" for d, f in fidelity_by_depth.items())
              + f" -> depth {tree_depth}")
    else:
        tree_depth = int(args.tree_depth)

    print(f"[{body_part}] fitting surrogate reasoning tree (depth={tree_depth}) ...")
    reasoning_tree = train_reasoning_tree(
        pca_features=train_components,
        main_model_predictions=train_main_predictions,
        max_depth=tree_depth,
    )

    # Fidelity = how often the simple tree agrees with the real model it's
    # meant to explain. Low fidelity means the explanations would be
    # misleading, so it's worth watching this number, not just accuracy.
    test_main_predictions = main_model.predict(test_components)
    tree_predictions_on_test = reasoning_tree.predict(test_components)
    fidelity = accuracy_score(test_main_predictions, tree_predictions_on_test)
    print(f"[{body_part}] surrogate tree fidelity to main model (test set): {fidelity:.1%}")
    if fidelity < 0.85:
        print(f"[{body_part}] WARNING: fidelity below 85% -- explanations may not "
              f"reliably reflect the main model's actual reasoning.")

    region_labels = label_components_by_region(pca)

    joblib.dump(pca, models_dir / f"{body_part}_pca.joblib")
    joblib.dump(main_model, models_dir / f"{body_part}_classifier.joblib")
    joblib.dump(reasoning_tree, models_dir / f"{body_part}_reasoning_tree.joblib")
    joblib.dump(region_labels, models_dir / f"{body_part}_region_labels.joblib")
    joblib.dump({
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "fidelity": float(fidelity),
        "n_components": int(n_components),
        "variance_retained": float(explained),
        "tree_depth": int(tree_depth),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_images": {"normal": int((labels == 0).sum()), "abnormal": int((labels == 1).sum())},
    }, models_dir / f"{body_part}_metrics.joblib")
    print(f"[{body_part}] saved 5 model artifacts -> {models_dir}")


if __name__ == "__main__":
    main()
