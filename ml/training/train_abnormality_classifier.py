import argparse
from pathlib import Path
import joblib,numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.model_selection import train_test_split
from ml.explain import label_components_by_region,train_reasoning_tree
from ml.preprocessing import batch_preprocess
from ml.training.dataset_utils import list_images_by_class
PROJECT_ROOT=Path(__file__).resolve().parents[2]
LABEL_TO_INT={"normal":0,"abnormal":1}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--body-part",required=True); p.add_argument("--datasets-root",default=str(PROJECT_ROOT/"datasets")); p.add_argument("--models-dir",default=str(PROJECT_ROOT/"ml"/"trained_models")); p.add_argument("--n-components",type=int,default=20); p.add_argument("--test-size",type=float,default=.2); a=p.parse_args(); out=Path(a.models_dir); out.mkdir(parents=True,exist_ok=True); paths,labels=list_images_by_class(Path(a.datasets_root),a.body_part); y=np.array([LABEL_TO_INT[x] for x in labels]); tr,te,ytr,yte=train_test_split(paths,y,test_size=a.test_size,stratify=y,random_state=42); Xtr,iv=batch_preprocess(tr); ytr=ytr[iv]; Xte,iv=batch_preprocess(te); yte=yte[iv]; n=min(a.n_components,Xtr.shape[0]-1,Xtr.shape[1]); pca=PCA(n_components=n,random_state=42); A=pca.fit_transform(Xtr); B=pca.transform(Xte); clf=LogisticRegression(max_iter=3000,class_weight="balanced"); clf.fit(A,ytr); pred=clf.predict(B); tree=train_reasoning_tree(A,clf.predict(A),max_depth=4); regions=label_components_by_region(pca); joblib.dump(pca,out/f"{a.body_part}_pca.joblib"); joblib.dump(clf,out/f"{a.body_part}_classifier.joblib"); joblib.dump(tree,out/f"{a.body_part}_reasoning_tree.joblib"); joblib.dump(regions,out/f"{a.body_part}_region_labels.joblib"); joblib.dump({"test_accuracy":float(accuracy_score(yte,pred)),"test_balanced_accuracy":float(balanced_accuracy_score(yte,pred)),"fidelity":float(accuracy_score(clf.predict(B),tree.predict(B))),"n_components":n,"variance_retained":float(pca.explained_variance_ratio_.sum()),"tree_depth":4,"n_train":len(ytr),"n_test":len(yte),"n_images":{"normal":int((y==0).sum()),"abnormal":int((y==1).sum())}},out/f"{a.body_part}_metrics.joblib")
if __name__=="__main__": main()
