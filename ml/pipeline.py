"""
Inference pipeline used by the Django view.

Two-stage design:
  1. Body-part router: which body part is this X-ray of? (chest / bone)
  2. Body-part-specific model: normal vs. abnormal for that body part,
     using its own PCA space and classifier (a chest PCA basis is
     meaningless for a hand X-ray and vice versa).

Each stage also carries a small surrogate decision tree (see explain.py)
used only to generate the human-readable reasoning text.

All model artifacts are loaded once, lazily, and cached in-process
(_MODEL_CACHE) so a Django dev/prod server doesn't hit disk on every
request. Nothing here is Django-specific except the settings import, so
the same module doubles as the entry point for offline testing.
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .explain import explanation_steps
from .preprocessing import IMAGE_SIZE, colorfulness, preprocess_pil_image

try:
    from django.conf import settings

    MODELS_DIR = Path(settings.ML_MODELS_DIR)
    SUPPORTED_BODY_PARTS = list(settings.SUPPORTED_BODY_PARTS)
except Exception:
    # Allows `python -m ml.pipeline` outside Django (e.g. quick manual test)
    MODELS_DIR = Path(__file__).resolve().parent / "trained_models"
    SUPPORTED_BODY_PARTS = ["chest", "bone"]

BODY_PART_ROUTER_PATH = MODELS_DIR / "body_part_router.joblib"
ROUTER_OOD_STATS_PATH = MODELS_DIR / "router_ood_stats.joblib"

ABNORMALITY_CLASS_NAMES = {0: "normal", 1: "abnormal"}

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class BodyPartModelBundle:
    body_part: str
    pca: object
    classifier: object
    reasoning_tree: object
    component_region_labels: list[str]
    class_names: dict[int, str] = field(default_factory=lambda: dict(ABNORMALITY_CLASS_NAMES))


@dataclass
class AnalysisResult:
    # "ok" | "not_xray" | "unsupported_body_part" | "models_not_trained"
    status: str
    body_part: str | None = None
    body_part_confidence: float | None = None
    body_part_probabilities: dict[str, float] = field(default_factory=dict)
    abnormality_label: str | None = None
    abnormality_confidence: float | None = None
    abnormality_probabilities: dict[str, float] = field(default_factory=dict)
    explanation: str | None = None
    # Exact decomposition of the main classifier's decision (logistic
    # regression on PCA components): each entry is one component's
    # coefficient x score. Positive pushes toward "abnormal".
    contributions: list[dict] = field(default_factory=list)
    explanation_steps: list[dict] = field(default_factory=list)  # decision-tree rule path
    reasoning_label: str | None = None  # what the surrogate tree itself concluded
    pca_components: int | None = None
    pca_variance_retained: float | None = None
    processed_image_png: str | None = None  # base64: the 128x128 image the model saw
    reconstruction_png: str | None = None  # base64: that image rebuilt from PCA only
    warnings: list[str] = field(default_factory=list)
    message: str | None = None  # user-facing note for non-"ok" statuses

    def to_dict(self) -> dict:
        return asdict(self)


# Uploads whose average colour spread exceeds this are treated as ordinary
# colour photos, not X-rays (X-rays and photos of films score ~0.0-0.05).
MAX_COLORFULNESS = 0.12
# Out-of-distribution gate (see train_body_part_router.py): an image is
# refused if the router's PCA components explain less than this fraction of
# its variation, or if it is nearly featureless. Between the reject line and
# the 1st percentile of real training X-rays, a caution is shown instead.
OOD_REJECT_EXPLAINED = 0.35
OOD_MIN_VARIANCE_FRACTION = 0.4  # of the lowest variance seen in training
# Router confidence below this gets a "not sure which body part" warning.
LOW_ROUTER_CONFIDENCE = 0.80


def _to_png_b64(vector: np.ndarray) -> str:
    arr = np.clip(vector.reshape(IMAGE_SIZE), 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_joblib(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def _get_body_part_router():
    if "router" not in _MODEL_CACHE:
        _MODEL_CACHE["router"] = _load_joblib(BODY_PART_ROUTER_PATH)
    return _MODEL_CACHE["router"]


def _get_body_part_bundle(body_part: str) -> BodyPartModelBundle | None:
    cache_key = f"bundle:{body_part}"
    if cache_key not in _MODEL_CACHE:
        pca = _load_joblib(MODELS_DIR / f"{body_part}_pca.joblib")
        clf = _load_joblib(MODELS_DIR / f"{body_part}_classifier.joblib")
        tree = _load_joblib(MODELS_DIR / f"{body_part}_reasoning_tree.joblib")
        region_labels = _load_joblib(MODELS_DIR / f"{body_part}_region_labels.joblib")

        if pca is None or clf is None or tree is None or region_labels is None:
            _MODEL_CACHE[cache_key] = None
        else:
            _MODEL_CACHE[cache_key] = BodyPartModelBundle(
                body_part=body_part,
                pca=pca,
                classifier=clf,
                reasoning_tree=tree,
                component_region_labels=region_labels,
            )
    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Call after (re)training so a running server picks up new models."""
    _MODEL_CACHE.clear()


def _contributions(bundle: BodyPartModelBundle, z: np.ndarray, label: str, top: int = 4):
    """
    For logistic regression, logit = intercept + sum_j coef_j * z_j exactly,
    so coef_j * z_j is how much component j moved this prediction. Returns the
    largest movers plus a sentence built from the ones that pushed toward the
    predicted label.
    """
    coef = bundle.classifier.coef_[0]
    raw = coef * z
    order = np.argsort(-np.abs(raw))[:top]
    total = float(np.abs(raw).sum()) or 1.0
    items = []
    for j in order:
        items.append({
            "component": int(j) + 1,
            "region": bundle.component_region_labels[j],
            "contribution": float(raw[j]),
            "share": float(abs(raw[j]) / total),
            "toward": "abnormal" if raw[j] > 0 else "normal",
        })
    supporting = [c for c in items if c["toward"] == label][:3]
    if supporting:
        regions = []
        for c in supporting:
            if c["region"] not in regions:
                regions.append(c["region"])
        where = ", ".join(regions[:-1]) + (" and " if len(regions) > 1 else "") + regions[-1]
        sentence = (
            f"The prediction '{label}' was driven mostly by patterns toward the {where} "
            f"of the image (components "
            + ", ".join(f"#{c['component']}" for c in supporting)
            + "). This describes which image patterns the model weighed, not a medical diagnosis."
        )
    else:
        sentence = f"No single pattern dominated; the '{label}' call came from many small effects."
    return items, sentence


def _get_ood_stats():
    if "ood" not in _MODEL_CACHE:
        _MODEL_CACHE["ood"] = _load_joblib(ROUTER_OOD_STATS_PATH)
    return _MODEL_CACHE["ood"]


def analyze_xray(image_file) -> AnalysisResult:
    """
    image_file: a path or a file-like object (e.g. a Django UploadedFile).
    """
    router = _get_body_part_router()
    if router is None:
        return AnalysisResult(
            status="models_not_trained",
            message=(
                "The body-part router hasn't been trained yet. Run the training "
                "scripts in ml/training/ once a dataset is available, then retry."
            ),
        )

    img = Image.open(image_file)
    img.load()

    if colorfulness(img) > MAX_COLORFULNESS:
        return AnalysisResult(
            status="not_xray",
            message=(
                "This looks like a colour photo rather than an X-ray. Please upload "
                "a chest or bone X-ray image (a scan, or a clear photo of the film)."
            ),
        )

    vector = preprocess_pil_image(img).reshape(1, -1)
    warnings: list[str] = []

    # Out-of-distribution check: how much of this image's variation can the
    # router's PCA components (learned from real X-rays) explain?
    router_pca = router.named_steps["pca"]
    recon_error = float(np.mean((vector - router_pca.inverse_transform(router_pca.transform(vector))) ** 2))
    variance = float(vector.var())
    explained = 1.0 - recon_error / max(variance, 1e-9)
    ood = _get_ood_stats()
    if ood and "explained_min" in ood:
        if (
            variance < OOD_MIN_VARIANCE_FRACTION * ood["variance_min"]
            or explained < OOD_REJECT_EXPLAINED
        ):
            return AnalysisResult(
                status="not_xray",
                processed_image_png=_to_png_b64(vector[0]),
                message=(
                    "This image doesn't resemble the chest or bone X-rays the models "
                    "were trained on: their principal components can only explain "
                    f"{max(explained, 0):.0%} of it (real X-rays: typically "
                    f"{ood['explained_median']:.0%}). No prediction is given."
                ),
            )
        if explained < ood["explained_p1"]:
            warnings.append(
                "This image is less typical than almost every X-ray in the training "
                "set, so treat the result with extra caution."
            )

    body_part_probs = router.predict_proba(vector)[0]
    body_part_idx = int(np.argmax(body_part_probs))
    body_part = str(router.classes_[body_part_idx])
    body_part_confidence = float(body_part_probs[body_part_idx])
    body_part_probabilities = {str(c): float(p) for c, p in zip(router.classes_, body_part_probs)}
    if body_part_confidence < LOW_ROUTER_CONFIDENCE:
        warnings.append(
            f"The model isn't sure which body part this is ({body_part_confidence:.0%} "
            f"confident it's {body_part})."
        )

    if body_part not in SUPPORTED_BODY_PARTS:
        return AnalysisResult(
            status="unsupported_body_part",
            body_part=body_part,
            body_part_confidence=body_part_confidence,
            body_part_probabilities=body_part_probabilities,
            message=(
                f"This looks like a {body_part} X-ray. Detailed analysis for that "
                f"body part isn't available yet -- currently supported: "
                f"{', '.join(SUPPORTED_BODY_PARTS)}."
            ),
        )

    bundle = _get_body_part_bundle(body_part)
    if bundle is None:
        return AnalysisResult(
            status="models_not_trained",
            body_part=body_part,
            body_part_confidence=body_part_confidence,
            body_part_probabilities=body_part_probabilities,
            message=(
                f"Identified as a {body_part} X-ray, but its abnormality model "
                f"hasn't been trained yet."
            ),
        )

    components = bundle.pca.transform(vector)
    abnormality_probs = bundle.classifier.predict_proba(components)[0]
    abnormality_idx = int(np.argmax(abnormality_probs))
    abnormality_label = bundle.class_names.get(abnormality_idx, str(abnormality_idx))
    abnormality_confidence = float(abnormality_probs[abnormality_idx])
    abnormality_probabilities = {
        bundle.class_names.get(int(c), str(c)): float(p)
        for c, p in zip(bundle.classifier.classes_, abnormality_probs)
    }

    steps, tree_label = explanation_steps(
        bundle.reasoning_tree, components[0], bundle.component_region_labels, bundle.class_names,
    )
    if tree_label != abnormality_label:
        warnings.append(
            f"The step-by-step reasoning model reaches '{tree_label}' on this image, "
            f"unlike the main model. Its rule path is shown for transparency, but the "
            f"'what drove this prediction' breakdown is the faithful account."
        )

    reconstruction = bundle.pca.inverse_transform(components)[0]
    contributions, explanation = _contributions(bundle, components[0], abnormality_label)

    return AnalysisResult(
        status="ok",
        body_part=body_part,
        body_part_confidence=body_part_confidence,
        body_part_probabilities=body_part_probabilities,
        abnormality_label=abnormality_label,
        abnormality_confidence=abnormality_confidence,
        abnormality_probabilities=abnormality_probabilities,
        explanation=explanation,
        contributions=contributions,
        explanation_steps=steps,
        reasoning_label=tree_label,
        pca_components=int(bundle.pca.n_components_),
        pca_variance_retained=float(bundle.pca.explained_variance_ratio_.sum()),
        processed_image_png=_to_png_b64(vector[0]),
        reconstruction_png=_to_png_b64(reconstruction),
        warnings=warnings,
    )


def model_info() -> dict:
    """Metrics saved at training time, for display in the UI."""
    if "info" not in _MODEL_CACHE:
        info = {"router": _load_joblib(MODELS_DIR / "router_metrics.joblib")}
        for part in SUPPORTED_BODY_PARTS:
            info[part] = _load_joblib(MODELS_DIR / f"{part}_metrics.joblib")
        _MODEL_CACHE["info"] = info
    return _MODEL_CACHE["info"]
