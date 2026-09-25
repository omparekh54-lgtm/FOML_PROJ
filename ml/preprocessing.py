from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
IMAGE_SIZE=(128,128)
def preprocess_pil_image(img:Image.Image)->np.ndarray:
    gray=ImageOps.grayscale(img)
    gray=ImageOps.fit(gray,IMAGE_SIZE,method=Image.Resampling.LANCZOS)
    return np.asarray(gray,dtype=np.float32)/255.0
def load_and_preprocess(image_path:str|Path)->np.ndarray:
    with Image.open(image_path) as img: return preprocess_pil_image(img)
def colorfulness(img:Image.Image)->float:
    rgb=img.convert("RGB")
    arr=np.asarray(rgb,dtype=np.float32)/255.0
    return float(np.mean(np.max(arr,axis=2)-np.min(arr,axis=2)))
