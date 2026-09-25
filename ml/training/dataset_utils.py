from pathlib import Path
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def _iter_images(folder):
 if not folder.is_dir(): return
 for p in sorted(folder.iterdir()):
  if p.suffix.lower() in IMAGE_EXTENSIONS: yield p
def list_images_by_bodypart(datasets_root):
 paths=[]; labels=[]
 for b in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
  for c in sorted(p for p in b.iterdir() if p.is_dir()):
   for f in _iter_images(c): paths.append(f); labels.append(b.name)
 return paths,labels
def list_images_by_class(datasets_root,body_part):
 root=datasets_root/body_part
 if not root.is_dir(): raise FileNotFoundError(f"No dataset folder at {root}.")
 paths=[]; labels=[]
 for c in sorted(p for p in root.iterdir() if p.is_dir()):
  for f in _iter_images(c): paths.append(f); labels.append(c.name)
 return paths,labels
