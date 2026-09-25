from __future__ import annotations
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
import joblib, numpy as np
from PIL import Image
from .preprocessing import IMAGE_SIZE, colorfulness, preprocess_pil_image
try:
    from django.conf import settings
    MODELS_DIR=Path(settings.ML_MODELS_DIR)
    SUPPORTED_BODY_PARTS=list(settings.SUPPORTED_BODY_PARTS)
except Exception:
    MODELS_DIR=Path(__file__).resolve().parent/"trained_models"; SUPPORTED_BODY_PARTS=["chest","bone"]
ABNORMALITY_CLASS_NAMES={0:"normal",1:"abnormal"}; _MODEL_CACHE={}
@dataclass
class AnalysisResult:
    status:str
    body_part:str|None=None
    body_part_confidence:float|None=None
    body_part_probabilities:dict[str,float]=field(default_factory=dict)
    abnormality_label:str|None=None
    abnormality_confidence:float|None=None
    abnormality_probabilities:dict[str,float]=field(default_factory=dict)
    explanation:str|None=None
    contributions:list[dict]=field(default_factory=list)
    explanation_steps:list[dict]=field(default_factory=list)
    reasoning_label:str|None=None
    pca_components:int|None=None
    pca_variance_retained:float|None=None
    processed_image_png:str|None=None
    reconstruction_png:str|None=None
    warnings:list[str]=field(default_factory=list)
    message:str|None=None
    def to_dict(self): return asdict(self)
def _load(path):
    return joblib.load(path) if path.exists() else None
def _router():
    if "router" not in _MODEL_CACHE: _MODEL_CACHE["router"]=_load(MODELS_DIR/"body_part_router.joblib")
    return _MODEL_CACHE["router"]
def _bundle(part):
    k=f"bundle:{part}"
    if k not in _MODEL_CACHE:
        _MODEL_CACHE[k]=None
        p,c,t,r=[_load(MODELS_DIR/f"{part}_{x}.joblib") for x in ("pca","classifier","reasoning_tree","region_labels")]
        if all(x is not None for x in (p,c,t,r)): _MODEL_CACHE[k]=(p,c,t,r)
    return _MODEL_CACHE[k]
def clear_model_cache(): _MODEL_CACHE.clear()
def _b64(arr):
    import base64
    img=Image.fromarray((np.clip(arr.reshape(IMAGE_SIZE),0,1)*255).astype("uint8"))
    b=io.BytesIO(); img.save(b,"PNG"); return base64.b64encode(b.getvalue()).decode()
def analyze_xray(image_file):
    router=_router()
    if router is None: return AnalysisResult("models_not_trained",message="Models are not trained yet. Run the training scripts.")
    img=Image.open(image_file); img.load()
    if colorfulness(img)>0.12: return AnalysisResult("not_xray",message="This looks like a colour photo rather than an X-ray.")
    x=preprocess_pil_image(img).reshape(1,-1)
    pca=router["pca"]; clf=router["classifier"]; z=pca.transform(x)
    probs=clf.predict_proba(z)[0] if hasattr(clf,"predict_proba") else np.zeros(len(clf.classes_))
    label=clf.classes_[int(np.argmax(probs))]; conf=float(np.max(probs)); part=str(label)
    if part not in SUPPORTED_BODY_PARTS: return AnalysisResult("unsupported_body_part",body_part=part,body_part_confidence=conf,message="This body part is not supported by the trained models.")
    bundle=_bundle(part)
    if bundle is None: return AnalysisResult("models_not_trained",body_part=part,body_part_confidence=conf,message=f"The {part} model is not trained yet.")
    bpca,bclf,tree,regions=bundle; zz=bpca.transform(x)
    ap=bclf.predict_proba(zz)[0] if hasattr(bclf,"predict_proba") else np.zeros(2)
    ai=int(np.argmax(ap)); abnormality=ABNORMALITY_CLASS_NAMES.get(int(bclf.classes_[ai]),str(bclf.classes_[ai])); ac=float(ap[ai])
    explanation=f"The {abnormality} prediction is based on patterns extracted from the X-ray in the model's PCA space. This is an experimental model output, not a medical diagnosis."
    return AnalysisResult("ok",part,conf,{str(k):float(v) for k,v in zip(clf.classes_,probs)},abnormality,ac,{"normal":float(ap[0]) if len(ap)>0 else 0.0,"abnormal":float(ap[-1]) if len(ap)>1 else 0.0},explanation,pca_components=getattr(bpca,"n_components_",None),pca_variance_retained=float(np.sum(getattr(bpca,"explained_variance_ratio_",[]))))
def model_info(): return {}
