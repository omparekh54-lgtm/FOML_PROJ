from pathlib import Path
import argparse, joblib, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.preprocessing import load_and_preprocess
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="datasets"); ap.add_argument("--out",default="ml/trained_models"); a=ap.parse_args(); X=[]; y=[]
    for part in ("chest","bone"):
        for f in Path(a.data,part).glob("*/*"):
            try: X.append(load_and_preprocess(f)); y.append(part)
            except Exception: pass
    X=np.asarray(X); X=StandardScaler().fit_transform(X); pca=PCA(n_components=min(64,X.shape[1]),random_state=42); Z=pca.fit_transform(X); clf=SVC(probability=True,random_state=42); clf.fit(Z,y)
    Path(a.out).mkdir(parents=True,exist_ok=True); joblib.dump({"pca":pca,"classifier":clf},Path(a.out)/"body_part_router.joblib")
if __name__=="__main__": main()
