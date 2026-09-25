from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
IMAGE_SIZE=(128,128)

def preprocess_pil_image(img):
    img=ImageOps.exif_transpose(img).convert("L")
    img=ImageOps.equalize(img).resize(IMAGE_SIZE,Image.LANCZOS)
    return (np.asarray(img,dtype=np.float32)/255.0).flatten()

def load_and_preprocess(image_path):
    with Image.open(image_path) as img: return preprocess_pil_image(img)

def colorfulness(img):
    rgb=ImageOps.exif_transpose(img).convert("RGB"); rgb.thumbnail((256,256))
    arr=np.asarray(rgb,dtype=np.float32)
    return float((arr.max(axis=2)-arr.min(axis=2)).mean()/255.0)

def batch_preprocess(image_paths):
    vectors=[]; valid=[]
    for i,p in enumerate(image_paths):
        try: vectors.append(load_and_preprocess(p)); valid.append(i)
        except Exception as exc: print(f"[preprocessing] skipped {p}: {exc}")
    return (np.vstack(vectors) if vectors else np.empty((0,IMAGE_SIZE[0]*IMAGE_SIZE[1]),dtype=np.float32),valid)
