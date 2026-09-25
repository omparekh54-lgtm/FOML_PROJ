from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from PIL import Image, UnidentifiedImageError
from ml.pipeline import analyze_xray, model_info

SAMPLES_DIR = Path(settings.BASE_DIR) / "analyzer" / "static" / "analyzer" / "samples"
SAMPLE_LABELS = {"chest_normal": "Chest · normal", "chest_abnormal": "Chest · pneumonia", "bone_normal": "Bone · no fracture", "bone_abnormal": "Bone · fracture"}

def _samples():
    samples=[]
    if SAMPLES_DIR.is_dir():
        for f in sorted(SAMPLES_DIR.glob("*.jpg")):
            samples.append({"file": f"analyzer/samples/{f.name}", "label": SAMPLE_LABELS.get(f.stem.rsplit("_",1)[0], f.stem)})
    return samples

def _pct(x): return f"{x:.0%}" if isinstance(x,(int,float)) else "—"
def _model_cards():
    info=model_info(); cards=[]; router=info.get("router")
    if router:
        cards.append({"name":"Body-part router","task":"Chest or bone?","headline":_pct(router["test_accuracy"]),"headline_label":"test accuracy","rows":[("Principal components",router["n_components"]),("Variance kept",_pct(router["variance_retained"])),("Classifier","RBF support-vector machine"),("Test images",router["n_test"]) ]})
    for part,task in (("chest","Normal vs. pneumonia"),("bone","Fractured vs. not")):
        m=info.get(part)
        if m: cards.append({"name":f"{part.title()} model","task":task,"headline":_pct(m["test_balanced_accuracy"]),"headline_label":"balanced test accuracy","rows":[("Principal components",m["n_components"]),("Variance kept",_pct(m["variance_retained"])),("Classifier","Logistic regression"),("Reasoning-tree fidelity",_pct(m["fidelity"])),("Training images",f"{m['n_images']['normal']} normal · {m['n_images']['abnormal']} abnormal")]})
    return cards

def index(request): return render(request,"analyzer/index.html",{"samples":_samples(),"model_cards":_model_cards(),"max_mb":settings.MAX_UPLOAD_SIZE_BYTES//(1024*1024)})

@require_POST
def api_analyze(request):
    upload=request.FILES.get("image")
    if upload is None: return JsonResponse({"status":"error","message":"No image was attached."},status=400)
    if upload.size > settings.MAX_UPLOAD_SIZE_BYTES: return JsonResponse({"status":"error","message":f"That file is larger than {settings.MAX_UPLOAD_SIZE_BYTES//(1024*1024)} MB."},status=400)
    try:
        with Image.open(upload) as probe: probe.verify()
        upload.seek(0); result=analyze_xray(upload)
    except (UnidentifiedImageError,OSError,ValueError): return JsonResponse({"status":"error","message":"That file couldn't be read as an image. Try a JPG or PNG."},status=400)
    return JsonResponse(result.to_dict())
