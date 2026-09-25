from pathlib import Path
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def list_images_by_bodypart(datasets_root):
    out={}
    for p in Path(datasets_root).iterdir() if Path(datasets_root).is_dir() else []:
        if p.is_dir(): out[p.name]=[x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTENSIONS]
    return out
