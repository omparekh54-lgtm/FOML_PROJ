import argparse
from pathlib import Path
import joblib,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ml.preprocessing import batch_preprocess
from ml.training.dataset_utils import list_images_by_bodypart
PROJECT_ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--datasets-root",default=str(PROJECT_ROOT/"datasets")); p.add_argument("--models-dir",default=str(PROJECT_ROOT/"ml"/"trained_models")); p.add_argument("--n-components",type=int,default=60); p.add_argument("--test-size",type=float,default=.2); a=p.parse_args(); root=Path(a.datasets_root); out=Path(a.models_dir); out.mkdir(parents=True,exist_ok=True)
 paths,labels=list_images_by_bodypart(root)
 if not paths: raise SystemExit(f"No images found under {root}.")
 tr,te,ytr,yte=train_test_split(paths,labels,test_size=a.test_size,stratify=labels,random_state=42); Xtr,iv=batch_preprocess(tr); ytr=[ytr[i] for i in iv]; Xte,iv=batch_preprocess(te); yte=[yte[i] for i in iv]
 n=min(a.n_components,Xtr.shape[0]-1,Xtr.shape[1]); pipe=Pipeline([("pca",PCA(n_components=n,random_state=42)),("scale",StandardScaler()),("clf",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42))]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); pca=pipe.named_steps["pca"]; joblib.dump(pipe,out/"body_part_router.joblib")
 allx=np.vstack([Xtr,Xte]); recon=pca.inverse_transform(pca.transform(allx)); err=np.mean((allx-recon)**2,axis=1); var=allx.var(axis=1); ex=1-err/np.maximum(var,1e-9); joblib.dump({"explained_min":float(ex.min()),"explained_p1":float(np.percentile(ex,1)),"explained_median":float(np.median(ex)),"variance_min":float(var.min())},out/"router_ood_stats.joblib"); joblib.dump({"test_accuracy":float(accuracy_score(yte,pred)),"n_components":n,"variance_retained":float(pca.explained_variance_ratio_.sum()),"n_train":len(ytr),"n_test":len(yte)},out/"router_metrics.joblib"); print(f"router accuracy: {accuracy_score(yte,pred):.1%}")
if __name__=="__main__": main()
