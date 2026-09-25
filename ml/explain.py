from __future__ import annotations
def explanation_steps(tree, feature_vector, feature_names):
    try:
        node=0; steps=[]
        while tree.children_left[node] != tree.children_right[node]:
            feat=tree.feature[node]; threshold=tree.threshold[node]
            direction=feature_vector[feat] <= threshold
            steps.append({"component":int(feat)+1,"feature":feature_names[feat],"threshold":float(threshold),"direction":"left" if direction else "right"})
            node=tree.children_left[node] if direction else tree.children_right[node]
        return steps
    except Exception:
        return []
