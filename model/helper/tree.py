# MODEL SUMMARY
def print_model_summary(model):
    tree = model.tree_

    print("\n" + "-" * 70)
    print("DECISION TREE")
    print("-" * 70)

    print(f"Total Nodes   : {tree.node_count}")
    print(f"Leaf Nodes    : {model.get_n_leaves()}")
    print(f"Tree Depth    : {model.get_depth()}")


# TREE RULES (EXPLANATION)
def print_tree_rules(tree, node=0, depth=0, max_depth=5):
    indent = "  " * depth

    value = tree.value[node][0]
    samples = tree.n_node_samples[node]

    class_label = "NORMAL" if value[0] > value[1] else "PNEUMONIA"
    confidence = max(value) / sum(value) * 100

    # Leaf node
    if tree.feature[node] == -2:
        print(f"{indent}→ LEAF: {class_label} | Confidence: {confidence:.1f}% | Samples: {samples}")
        return

    # Depth limit
    if depth >= max_depth:
        print(f"{indent}→ STOP: {class_label} | Confidence: {confidence:.1f}% | Samples: {samples}")
        return

    feature = tree.feature[node]
    threshold = tree.threshold[node]

    print(f"{indent}[Node] feature_{feature} <= {threshold:.4f}")
    print(f"{indent}Samples: {samples} | Class: {class_label} | Confidence: {confidence:.1f}%")

    print(f"{indent}→ TRUE:")
    print_tree_rules(tree, tree.children_left[node], depth + 1, max_depth)

    print(f"{indent}→ FALSE:")
    print_tree_rules(tree, tree.children_right[node], depth + 1, max_depth)


# CONFUSION MATRIX
def print_confusion_matrix(cm):
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print("                 Predicted")
    print("              NORMAL  PNEUMONIA")
    print(f"Actual NORMAL   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"       PNEUMONIA {cm[1,0]:6d}   {cm[1,1]:6d}")


# METRICS
def print_metrics(acc, precision, recall, f1):
    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE")
    print("-" * 70)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")