from pathlib import Path
import argparse, joblib, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ml.preprocessing import load_and_preprocess
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--body-part",required=True); ap.add_argument("--data",default="datasets"); ap.add_argument("--out",default="ml/trained_models"); a=ap.parse_args(); X=[]; y=[]
    for idx,label in enumerate(("normal","abnormal")):
        for f in Path(a.data,a.body_part,label).glob("*"):
            try: X.append(load_and_preprocess(f)); y.append(idx)
            except Exception: pass
    X=np.asarray(X); pca=PCA(n_components=min(32,max(2,X.shape[0]-1)),random_state=42).fit(X); Z=pca.transform(X); clf=LogisticRegression(max_iter=2000).fit(Z,y); tree=DecisionTreeClassifier(max_depth=4,random_state=42).fit(Z,clf.predict(Z)); labels=["upper-left","upper-center","upper-right","mid-left","center","mid-right","lower-left","lower-center","lower-right"][:Z.shape[1]]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); joblib.dump(pca,out/f"{a.body_part}_pca.joblib"); joblib.dump(clf,out/f"{a.body_part}_classifier.joblib"); joblib.dump(tree,out/f"{a.body_part}_reasoning_tree.joblib"); joblib.dump(labels,out/f"{a.body_part}_region_labels.joblib")
if __name__=="__main__": main()
