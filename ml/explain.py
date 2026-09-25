"""
The "reasoning" layer.

The main abnormality classifier (Logistic Regression / SVM on top of PCA
components) is accurate but not something you can read out loud to a user —
"class 1 with probability 0.83" means nothing to a patient.

So we train a second, much simpler model on the *same* PCA-reduced features:
a shallow decision tree that mimics the main classifier's decisions (a
"surrogate model", a standard explainability technique). A decision tree's
prediction is just a path of if/else rules, e.g.

    PC_2 <= -0.41  ->  PC_5 > 0.18  ->  predict "abnormal"

which we can walk and translate into a sentence. That sentence is the
"logical reasoning" shown to the user alongside the main model's prediction.

To make the sentence mean something (not "PC_5 is high"), each principal
component is tagged with the image region it responds to most strongly,
computed once from its pixel loadings (pca.components_) at training time.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from .preprocessing import IMAGE_SIZE

# A simple 3x3 naming grid. PCA on X-rays doesn't respect anatomical
# boundaries, so these are coarse, best-effort location hints, not medical
# regions -- the UI copy should always hedge them as "pattern located
# toward the ___ of the image", never as a clinical finding.
_GRID_LABELS = [
    "upper-left", "upper-center", "upper-right",
    "mid-left", "center", "mid-right",
    "lower-left", "lower-center", "lower-right",
]


def label_components_by_region(pca, image_size: tuple[int, int] = IMAGE_SIZE) -> list[str]:
    """
    For each principal component, find which 3x3 grid cell of the image it
    loads onto most strongly (by summed absolute pixel weight) and return a
    human-readable label like "upper-right" for that component's index.
    """
    h, w = image_size
    row_edges = [0, h // 3, 2 * h // 3, h]
    col_edges = [0, w // 3, 2 * w // 3, w]

    labels = []
    for component in pca.components_:
        grid = component.reshape(h, w)
        cell_scores = []
        for r in range(3):
            for c in range(3):
                cell = grid[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
                cell_scores.append(np.abs(cell).sum())
        labels.append(_GRID_LABELS[int(np.argmax(cell_scores))])
    return labels


def train_reasoning_tree(
    pca_features: np.ndarray,
    main_model_predictions: np.ndarray,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
) -> DecisionTreeClassifier:
    """
    Fit the surrogate decision tree on PCA features, using the MAIN model's
    predicted labels (not the ground-truth labels) as the target. This is
    what makes it a surrogate/explainer for that specific model rather than
    just a second, independent, weaker classifier.

    max_depth is kept small (3-4) on purpose: a deep tree gets more accurate
    at mimicking the main model but produces a rule path too long to read
    out as one or two sentences. min_samples_leaf guards against the same
    problem from the other direction -- on a small dataset, an unconstrained
    tree can carve out a leaf for a single quirky training point, which
    raises training-set fidelity while doing nothing (or worse) for held-out
    fidelity. Requiring a few samples per leaf keeps splits that generalize.
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
    )
    tree.fit(pca_features, main_model_predictions)
    return tree


def explanation_steps(
    tree: DecisionTreeClassifier,
    component_vector: np.ndarray,
    component_region_labels: list[str],
    class_names: dict[int, str],
) -> tuple[list[dict], str]:
    """
    Walk the decision path the surrogate tree took for one sample.

    Returns (steps, tree_label). Each step is a dict:
        {"component": 1-based PCA component number,
         "region": e.g. "upper-left",
         "direction": "higher" | "lower",
         "value": the sample's score on that component,
         "threshold": the split point the tree compared it against}

    A tree can test the same component twice on one path (e.g. "PC2 > -1"
    then "PC2 > 0.5"); only the last test of each component is kept, since
    it is the tighter bound and repeating the same region reads as noise.
    """
    t = tree.tree_
    node = 0
    by_component: dict[int, dict] = {}
    order: list[int] = []

    while t.feature[node] != _tree.TREE_UNDEFINED:
        feature_idx = int(t.feature[node])
        threshold = float(t.threshold[node])
        value = float(component_vector[feature_idx])
        region = (
            component_region_labels[feature_idx]
            if feature_idx < len(component_region_labels)
            else "an unspecified region"
        )
        goes_left = value <= threshold
        if feature_idx not in by_component:
            order.append(feature_idx)
        by_component[feature_idx] = {
            "component": feature_idx + 1,
            "region": region,
            "direction": "lower" if goes_left else "higher",
            "value": value,
            "threshold": threshold,
        }
        node = t.children_left[node] if goes_left else t.children_right[node]

    predicted_class = int(np.argmax(t.value[node]))
    tree_label = class_names.get(predicted_class, str(predicted_class))
    return [by_component[i] for i in order], tree_label


def explain_prediction(
    tree: DecisionTreeClassifier,
    component_vector: np.ndarray,
    component_region_labels: list[str],
    class_names: dict[int, str],
) -> str:
    """
    Turn the surrogate tree's decision path for one sample into a
    plain-language paragraph.

    component_vector: shape (n_components,) - the PCA projection of one image.
    """
    steps, label = explanation_steps(tree, component_vector, component_region_labels, class_names)

    if not steps:
        return f"The reasoning model reached '{label}' directly, with no strong distinguishing pattern."

    reasoning = "; then ".join(
        f"the pattern strength around the {s['region']} of the image was "
        f"{s['direction']} than typical (component #{s['component']})"
        for s in steps
    )
    return (
        f"The reasoning model leans toward '{label}' because {reasoning}. "
        f"This describes which patterns the model weighed, not a medical diagnosis."
    )
