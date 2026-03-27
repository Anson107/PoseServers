import json
import pickle
import sys
from pathlib import Path


def tree_depth(node, children_left, children_right):
    if children_left[node] == -1 and children_right[node] == -1:
        return 1
    return 1 + max(
        tree_depth(children_left[node], children_left, children_right),
        tree_depth(children_right[node], children_left, children_right),
    )


def convert_node(node_id, children_left, children_right, feature, threshold, value, depth=0):
    left_id = int(children_left[node_id])
    right_id = int(children_right[node_id])

    # value shape: [n_nodes, 1, 1]
    node_value = float(value[node_id][0][0])

    if left_id == -1 and right_id == -1:
        return {
            "type": "leaf",
            "value": node_value,
            "count": 1,
            "depth": depth,
        }

    return {
        "type": "node",
        "feature": int(feature[node_id]),
        "threshold": float(threshold[node_id]),
        "gain": 0.0,
        "value": node_value,
        "depth": depth,
        "left": convert_node(left_id, children_left, children_right, feature, threshold, value, depth + 1),
        "right": convert_node(right_id, children_left, children_right, feature, threshold, value, depth + 1),
    }


def main():
    if len(sys.argv) < 3:
        print("usage: convert_sklearn_pkl.py <input.pkl> <output.json>", file=sys.stderr)
        sys.exit(2)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    with in_path.open("rb") as f:
        model = pickle.load(f)

    if isinstance(model, dict) and isinstance(model.get("trees"), list):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
        return

    if not hasattr(model, "estimators_"):
        raise RuntimeError("Unsupported model: no estimators_ found")

    trees = []
    tree_metrics = []

    for idx, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        node = convert_node(0, children_left, children_right, feature, threshold, value)
        depth = tree_depth(0, children_left, children_right)
        trees.append(node)
        tree_metrics.append({"id": idx, "mae": 0.0, "depth": int(depth)})

    importance = []
    if hasattr(model, "feature_importances_"):
        importance = [float(x) for x in model.feature_importances_]

    max_depth = 1
    if tree_metrics:
        max_depth = max(t["depth"] for t in tree_metrics)

    out = {
        "version": 1,
        "algorithm": "sklearn-rf-regression-imported",
        "params": {
            "trees": int(getattr(model, "n_estimators", len(trees))),
            "maxDepth": int(getattr(model, "max_depth", max_depth) or max_depth),
            "minLeaf": int(getattr(model, "min_samples_leaf", 1)),
        },
        "trees": trees,
        "importance": importance,
        "metrics": {
            "mae": 0.0,
            "minMae": 0.0,
            "maxMae": 0.0,
            "avgDepth": float(sum(t["depth"] for t in tree_metrics) / len(tree_metrics)) if tree_metrics else 0.0,
            "maxDepth": max_depth,
        },
        "treeMetrics": tree_metrics,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
