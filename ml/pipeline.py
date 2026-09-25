import base64,io
from dataclasses import asdict,dataclass,field
from pathlib import Path
import joblib,numpy as np
from PIL import Image
from .explain import explanation_steps
from .preprocessing import IMAGE_SIZE,colorfulness,preprocess_pil_image
try:
 from django.conf import settings
 MODELS_DIR=Path(settings.ML_MODELS_DIR); SUPPORTED_BODY_PARTS=list(settings.SUPPORTED_BODY_PARTS)
except Exception:
 MODELS_DIR=Path(__file__).resolve().parent/"trained_models"; SUPPORTED_BODY_PARTS=["chest","bone"]
BODY_PART_ROUTER_PATH=MODELS_DIR/"body_part_router.joblib"; OOD_STATS=MODELS_DIR/"router_ood_stats.joblib"; ABNORMALITY_CLASS_NAMES={0:"normal",1:"abnormal"}; _MODEL_CACHE={}
@dataclass
class AnalysisResult:
 status:str; body_part:str|None=None; body_part_confidence:float|None=None; body_part_probabilities:dict=field(default_factory=dict); abnormality_label:str|None=None; abnormality_confidence:float|None=None; abnormality_probabilities:dict=field(default_factory=dict); explanation:str|None=None; contributions:list=field(default_factory=list); explanation_steps:list=field(default_factory=list); reasoning_label:str|None=None; pca_components:int|None=None; pca_variance_retained:float|None=None; processed_image_png:str|None=None; reconstruction_png:str|None=None; warnings:list=field(default_factory=list); message:str|None=None
 def to_dict(self): return asdict(self)
def _png(v):
 arr=np.clip(v.reshape(IMAGE_SIZE),0,1); b=io.BytesIO(); Image.fromarray((arr*255).astype(np.uint8),mode="L").save(b,"PNG"); return base64.b64encode(b.getvalue()).decode()
def _load(path): return joblib.load(path) if path.exists() else None
def _router():
 if "router" not in _MODEL_CACHE:_MODEL_CACHE["router"]=_load(BODY_PART_ROUTER_PATH)
 return _MODEL_CACHE["router"]
def _bundle(part):
 k=f"b:{part}"
 if k not in _MODEL_CACHE:
  p=_load(MODELS_DIR/f"{part}_pca.joblib"); c=_load(MODELS_DIR/f"{part}_classifier.joblib"); t=_load(MODELS_DIR/f"{part}_reasoning_tree.joblib"); r=_load(MODELS_DIR/f"{part}_region_labels.joblib"); _MODEL_CACHE[k]=(p,c,t,r) if all(x is not None for x in (p,c,t,r)) else None
 return _MODEL_CACHE[k]
def clear_model_cache(): _MODEL_CACHE.clear()
def analyze_xray(image_file):
 router=_router()
 if router is None:return AnalysisResult("models_not_trained",message="The body-part router hasn't been trained yet. Run the training scripts in ml/training/.")
 img=Image.open(image_file); img.load()
 if colorfulness(img)>0.12:return AnalysisResult("not_xray",message="This looks like a colour photo rather than an X-ray. Please upload a chest or bone X-ray image.")
 v=preprocess_pil_image(img); X=v.reshape(1,-1); warnings=[]
 probs=router.predict_proba(X)[0]; idx=int(np.argmax(probs)); part=str(router.classes_[idx]); conf=float(probs[idx]); bp={str(c):float(p) for c,p in zip(router.classes_,probs)}
 if conf<.80:warnings.append(f"The model isn't sure which body part this is ({conf:.0%} confident it's {part}).")
 if part not in SUPPORTED_BODY_PARTS:return AnalysisResult("unsupported_body_part",part,conf,bp,message=f"This looks like a {part} X-ray. Detailed analysis is not available yet.")
 bundle=_bundle(part)
 if bundle is None:return AnalysisResult("models_not_trained",part,conf,bp,message=f"Identified as a {part} X-ray, but its abnormality model hasn't been trained yet.")
 pca,clf,tree,regions=bundle; z=pca.transform(X); ap=clf.predict_proba(z)[0]; ai=int(np.argmax(ap)); label=ABNORMALITY_CLASS_NAMES.get(ai,str(ai)); ac=float(ap[ai]); apd={ABNORMALITY_CLASS_NAMES.get(int(c),str(c)):float(p) for c,p in zip(clf.classes_,ap)}; steps,tree_label=explanation_steps(tree,z[0],regions,ABNORMALITY_CLASS_NAMES)
 if tree_label!=label:warnings.append(f"The reasoning model reaches '{tree_label}', unlike the main model.")
 coef=clf.coef_[0]; raw=coef*z[0]; order=np.argsort(-np.abs(raw))[:4]; contrib=[{"component":int(j)+1,"region":regions[j],"contribution":float(raw[j]),"toward":"abnormal" if raw[j]>0 else "normal"} for j in order]
 supporting=[x for x in contrib if x["toward"]==label]; regions_text=", ".join(x["region"] for x in supporting[:3]) if supporting else "several image patterns"; explanation=f"The prediction '{label}' was driven mostly by patterns toward the {regions_text} of the image. This describes which image patterns the model weighed, not a medical diagnosis."
 recon=pca.inverse_transform(z)[0]
 return AnalysisResult("ok",part,conf,bp,label,ac,apd,explanation,contrib,steps,tree_label,int(pca.n_components_),float(pca.explained_variance_ratio_.sum()),_png(v),_png(recon),warnings)
def model_info():
 if "info" not in _MODEL_CACHE:
  _MODEL_CACHE["info"]={"router":_load(MODELS_DIR/"router_metrics.joblib")}
  for p in SUPPORTED_BODY_PARTS:_MODEL_CACHE["info"][p]=_load(MODELS_DIR/f"{p}_metrics.joblib")
 return _MODEL_CACHE["info"]
