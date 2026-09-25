import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from .preprocessing import IMAGE_SIZE
_GRID_LABELS=["upper-left","upper-center","upper-right","mid-left","center","mid-right","lower-left","lower-center","lower-right"]

def label_components_by_region(pca,image_size=IMAGE_SIZE):
    h,w=image_size; re=[0,h//3,2*h//3,h]; ce=[0,w//3,2*w//3,w]; labels=[]
    for component in pca.components_:
        grid=component.reshape(h,w); scores=[]
        for r in range(3):
            for c in range(3): scores.append(np.abs(grid[re[r]:re[r+1],ce[c]:ce[c+1]]).sum())
        labels.append(_GRID_LABELS[int(np.argmax(scores))])
    return labels

def train_reasoning_tree(pca_features,main_model_predictions,max_depth=3,min_samples_leaf=5):
    t=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,random_state=42); t.fit(pca_features,main_model_predictions); return t

def explanation_steps(tree,component_vector,component_region_labels,class_names):
    t=tree.tree_; node=0; by={}; order=[]
    while t.feature[node]!=_tree.TREE_UNDEFINED:
        idx=int(t.feature[node]); threshold=float(t.threshold[node]); value=float(component_vector[idx]); region=component_region_labels[idx] if idx<len(component_region_labels) else "an unspecified region"; left=value<=threshold
        if idx not in by: order.append(idx)
        by[idx]={"component":idx+1,"region":region,"direction":"lower" if left else "higher","value":value,"threshold":threshold}
        node=t.children_left[node] if left else t.children_right[node]
    return [by[i] for i in order],class_names.get(int(np.argmax(t.value[node])),str(int(np.argmax(t.value[node]))))

def explain_prediction(tree,component_vector,component_region_labels,class_names):
    steps,label=explanation_steps(tree,component_vector,component_region_labels,class_names)
    if not steps: return f"The reasoning model reached '{label}' directly, with no strong distinguishing pattern."
    reasoning="; then ".join(f"the pattern strength around the {s['region']} of the image was {s['direction']} than typical (component #{s['component']})" for s in steps)
    return f"The reasoning model leans toward '{label}' because {reasoning}. This describes which patterns the model weighed, not a medical diagnosis."
