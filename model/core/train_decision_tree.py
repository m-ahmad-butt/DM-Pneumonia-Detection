"""
Decision Tree Training on CNN Features
Shows detailed steps in terminal for demonstration
"""

import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

def print_header(title):
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def print_tree_rules(tree, node=0, depth=0, max_depth=10):
    """Recursively print decision tree rules, always ending with a decision."""
    if node == -1:
        return

    indent = "  " * depth
    samples = tree.n_node_samples[node]
    value = tree.value[node][0]
    gini = 1 - sum((v / sum(value)) ** 2 for v in value)
    class_label = "NORMAL" if value[0] > value[1] else "PNEUMONIA"
    confidence = max(value) / sum(value) * 100

    # Real leaf node
    if tree.feature[node] == -2:
        print(f"{indent}  +-- LEAF {node} -> DECISION: {class_label} | Confidence: {confidence:.1f}% | Samples: {samples}")
        return

    # Depth limit reached -- show best guess at this point
    if depth == max_depth:
        print(f"{indent}  +-- [DEPTH LIMIT] -> DECISION: {class_label} | Confidence: {confidence:.1f}% | Samples: {samples} | Gini: {gini:.4f}")
        return

    # Internal node -- print split info and recurse
    feature_idx = tree.feature[node]
    threshold = tree.threshold[node]
    normal_pct = value[0] / sum(value) * 100
    pneumonia_pct = value[1] / sum(value) * 100

    print(f"{indent}+-- Node {node} | Depth {depth} | Samples: {samples} | Gini: {gini:.4f}")
    print(f"{indent}|   Distribution -> NORMAL: {value[0]:.0f} ({normal_pct:.1f}%)  PNEUMONIA: {value[1]:.0f} ({pneumonia_pct:.1f}%)")
    print(f"{indent}|   Split: feature_{feature_idx} <= {threshold:.4f}")
    print(f"{indent}|")
    print(f"{indent}+-- [TRUE]  feature_{feature_idx} <= {threshold:.4f}")
    print_tree_rules(tree, tree.children_left[node],  depth + 1, max_depth)
    print(f"{indent}+-- [FALSE] feature_{feature_idx} > {threshold:.4f}")
    print_tree_rules(tree, tree.children_right[node], depth + 1, max_depth)


def collect_all_nodes(tree):
    """Collect all internal + leaf nodes with their decision info."""
    nodes = []
    for node_id in range(tree.node_count):
        value = tree.value[node_id][0]
        samples = tree.n_node_samples[node_id]
        class_label = "NORMAL" if value[0] > value[1] else "PNEUMONIA"
        confidence = max(value) / sum(value) * 100
        gini = 1 - sum((v / sum(value)) ** 2 for v in value)
        is_leaf = tree.feature[node_id] == -2

        nodes.append({
            "node_id"    : node_id,
            "type"       : "LEAF" if is_leaf else "NODE",
            "class_label": class_label,
            "confidence" : confidence,
            "samples"    : samples,
            "normal"     : int(value[0]),
            "pneumonia"  : int(value[1]),
            "gini"       : gini,
            "feature"    : tree.feature[node_id] if not is_leaf else None,
            "threshold"  : tree.threshold[node_id] if not is_leaf else None,
        })
    return nodes


def print_node_table(nodes, title):
    """Print a formatted table of nodes."""
    print(f"\n  {title}")
    print(f"  {'-' * 90}")
    print(f"  {'#':<5} {'Node':<8} {'Type':<6} {'Decision':<12} {'Confidence':>12} {'Normal':>8} {'Pneumonia':>10} {'Samples':>8} {'Gini':>8}")
    print(f"  {'-' * 90}")

    for i, n in enumerate(nodes, 1):
        print(
            f"  {i:<5} {n['node_id']:<8} {n['type']:<6} {n['class_label']:<12} "
            f"{n['confidence']:>11.1f}% {n['normal']:>8} {n['pneumonia']:>10} "
            f"{n['samples']:>8} {n['gini']:>8.4f}"
        )
    print(f"  {'-' * 90}")


# ------------------------------------------------------------------------------
FEATURES_DIR = Path(__file__).parent.parent / 'extracted_features'
MODEL_PATH   = Path(__file__).parent.parent / 'decision_tree_model.pkl'

# STEP 1 - LOAD DATA
print_header("STEP 1: LOADING CNN FEATURES")
X_train = np.load(FEATURES_DIR / 'X_train_features.npy')
y_train = np.load(FEATURES_DIR / 'y_train.npy')
X_test  = np.load(FEATURES_DIR / 'X_test_features.npy')
y_test  = np.load(FEATURES_DIR / 'y_test.npy')

print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Features/sample  : 1,280  (MobileNetV2 CNN)")
print(f"\n  Class distribution -- Training:")
print(f"    NORMAL    : {np.sum(y_train == 0):>5}")
print(f"    PNEUMONIA : {np.sum(y_train == 1):>5}")
print(f"\n  Class distribution -- Test:")
print(f"    NORMAL    : {np.sum(y_test == 0):>5}")
print(f"    PNEUMONIA : {np.sum(y_test == 1):>5}")

baseline_acc = max(np.sum(y_test == 0), np.sum(y_test == 1)) / len(y_test)
print(f"\n  Baseline accuracy (majority class): {baseline_acc:.4f}  ({baseline_acc*100:.2f}%)")

# STEP 2 - BUILD DECISION TREE
print_header("STEP 2: BUILDING DECISION TREE")
print("  Hyperparameters:")
print("    criterion        : gini")
print("    max_depth        : 10")
print("    min_samples_split: 20")
print("    min_samples_leaf : 10")
print("    random_state     : 42")

dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# STEP 3 - TRAIN
print_header("STEP 3: TRAINING DECISION TREE")
print("  Training in progress...")
dt_classifier.fit(X_train, y_train)
print("  Training completed!")

tree = dt_classifier.tree_
print(f"\n  Tree Structure:")
print(f"    Total nodes    : {tree.node_count}")
print(f"    Leaf nodes     : {dt_classifier.get_n_leaves()}")
print(f"    Internal nodes : {tree.node_count - dt_classifier.get_n_leaves()}")
print(f"    Tree depth     : {dt_classifier.get_depth()}")

# STEP 4 - DECISION RULES (FULL TREE, ALWAYS ENDS WITH DECISION)
print_header("STEP 4: DECISION RULES -- FULL TREE (MAX DEPTH 10)")
print_tree_rules(tree, max_depth=10)

# STEP 5 - FIRST 10 & LAST 10 NODES WITH DECISIONS
print_header("STEP 5: NODE SUMMARY -- FIRST 10 & LAST 10")
all_nodes = collect_all_nodes(tree)
print_node_table(all_nodes[:10],  "FIRST 10 NODES (from root side):")
print_node_table(all_nodes[-10:], "LAST 10 NODES (from leaf side):")

# STEP 6 - EVALUATE
print_header("STEP 6: MODEL EVALUATION")
y_pred = dt_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\n  Confusion Matrix:")
print("                    Predicted")
print("                 NORMAL   PNEUMONIA")
print(f"  Actual NORMAL    {cm[0,0]:5d}      {cm[0,1]:5d}")
print(f"         PNEUMONIA {cm[1,0]:5d}      {cm[1,1]:5d}")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n  Performance Metrics:")
print(f"    Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"    Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"    Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"    F1-Score  : {f1:.4f}")

# STEP 7 - SAVE
print_header("STEP 7: SAVING MODEL")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(dt_classifier, f)
print(f"  Model saved -> {MODEL_PATH}")

print("\n" + "-" * 80)
print("  DECISION TREE TRAINING COMPLETED!")
print("-" * 80 + "\n")