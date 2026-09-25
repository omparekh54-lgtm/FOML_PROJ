from pathlib import Path
import argparse, hashlib, random, shutil, re
from PIL import Image
PROJECT_ROOT=Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
AUGMENTED=re.compile(r"(-rotated\d)|( - Copy)|(\(\d+\))",re.I)
def _images(folder): return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if Path(folder).is_dir() else []
def _readable(p):
    try:
        with Image.open(p) as im: im.load()
        return True
    except Exception: return False
def _clean(paths):
    seen=set(); out=[]
    for p in paths:
        if AUGMENTED.search(p.stem): continue
        h=hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen and _readable(p): seen.add(h); out.append(p)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",default=str(PROJECT_ROOT)); ap.add_argument("--dest",default=str(PROJECT_ROOT/"datasets")); a=ap.parse_args()
    src,dst=Path(a.source),Path(a.dest); shutil.rmtree(dst,ignore_errors=True)
    for part,raw in (("chest",[("normal",src/"chest_xray/train/NORMAL"),("abnormal",src/"chest_xray/train/PNEUMONIA")]),("bone",[("normal",src/"Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/not fractured"),("abnormal",src/"Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/fractured")])):
        for label,folder in raw:
            files=_clean(_images(folder))[:450]; (dst/part/label).mkdir(parents=True,exist_ok=True)
            for f in files: shutil.copy2(f,dst/part/label/f.name.replace(" ","_"))
if __name__=="__main__": main()
