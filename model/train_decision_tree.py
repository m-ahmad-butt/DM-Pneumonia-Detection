import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve
)
import joblib
from decision_tree_classifier import DecisionTreeClassifier

# CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = str(SCRIPT_DIR / 'extracted_features')
MODELS_DIR = str(SCRIPT_DIR / 'hybrid_models')
RESULTS_DIR = str(SCRIPT_DIR / 'comparison_results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

print(f"\n{'-'*60}")
print("DECISION TREE CLASSIFIER ON CNN FEATURES")
print(f"{'-'*60}")

# Load extracted features
print(f"\n Loading CNN features...")
X_train = np.load(os.path.join(FEATURES_DIR, 'X_train_features.npy'))
y_train = np.load(os.path.join(FEATURES_DIR, 'y_train.npy'))
X_test = np.load(os.path.join(FEATURES_DIR, 'X_test_features.npy'))
y_test = np.load(os.path.join(FEATURES_DIR, 'y_test.npy'))

print(f"   Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"   Test: {X_test.shape}, Labels: {y_test.shape}")
print(f"   Feature dimension: {X_train.shape[1]}")

# Class distribution
print(f"\n  Class distribution:")
print(f"    NORMAL: {np.sum(y_train == 0)} train, {np.sum(y_test == 0)} test")
print(f"    PNEUMONIA: {np.sum(y_train == 1)} train, {np.sum(y_test == 1)} test")


# Train Decision Tree with class weights
print(f"\n Training Decision Tree...")

# Train with optimal hyperparameters
dt_classifier = DecisionTreeClassifier(
    max_depth=10,              # Prevent overfitting
    min_samples_split=20,      # Minimum samples to split
    min_samples_leaf=10,       # Minimum samples in leaf
    use_class_weights=True,    # Handle imbalance
    random_state=42
)

dt_classifier.fit(X_train, y_train)

# Print class weights
print(f"  Class weights: NORMAL={dt_classifier.class_weight_dict[0]:.3f}, PNEUMONIA={dt_classifier.class_weight_dict[1]:.3f}")
print(f"   Decision Tree trained")
print(f"   Tree depth: {dt_classifier.get_depth()}")
print(f"   Number of leaves: {dt_classifier.get_n_leaves()}")


# Save model
model_path = os.path.join(MODELS_DIR, 'decision_tree_classifier.pkl')
joblib.dump(dt_classifier, model_path)
print(f"   Model saved: {model_path}")


# Predictions
print(f"\n Making predictions...")
y_train_pred = dt_classifier.predict(X_train)
y_test_pred = dt_classifier.predict(X_test)
y_test_proba = dt_classifier.predict_proba(X_test)[:, 1]

print(f"   Predictions complete")


# Evaluation
print(f"\n Evaluating performance...")

print(f"\n{'─'*60}")
print("TRAINING SET PERFORMANCE")
print(f"{'─'*60}")
print(classification_report(y_train, y_train_pred, target_names=CLASS_NAMES))

print(f"\n{'─'*60}")
print("TEST SET PERFORMANCE")
print(f"{'─'*60}")
print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

# Per-class recall
normal_recall = recall_score(y_test, y_test_pred, pos_label=0)
pneumonia_recall = recall_score(y_test, y_test_pred, pos_label=1)

print(f"\n{'─'*60}")
print("KEY METRICS")
print(f"{'─'*60}")
print(f"  Accuracy:          {test_accuracy:.4f}")
print(f"  Precision:         {test_precision:.4f}")
print(f"  Recall (Overall):  {test_recall:.4f}")
print(f"  F1-Score:          {test_f1:.4f}")
print(f"  AUC-ROC:           {test_auc:.4f}")
print(f"\n  NORMAL Recall:     {normal_recall:.4f}")
print(f"  PNEUMONIA Recall:  {pneumonia_recall:.4f}")


# Save metrics
metrics = {
    'model': 'Decision Tree',
    'accuracy': test_accuracy,
    'precision': test_precision,
    'recall': test_recall,
    'f1_score': test_f1,
    'auc_roc': test_auc,
    'normal_recall': normal_recall,
    'pneumonia_recall': pneumonia_recall
}
np.save(os.path.join(RESULTS_DIR, 'dt_metrics.npy'), metrics)


# Visualizations
print(f"\n Creating visualizations...")

# 1. Decision Tree Structure (simplified to depth 3)
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
plot_tree(dt_classifier.model, max_depth=3, feature_names=[f'F{i}' for i in range(X_train.shape[1])],
          class_names=CLASS_NAMES, filled=True, ax=ax, fontsize=10)
ax.set_title('Decision Tree Structure (depth=3)', fontsize=16, fontweight='bold')
plt.tight_layout()
tree_path = os.path.join(RESULTS_DIR, 'dt_tree_structure.png')
plt.savefig(tree_path, dpi=150, bbox_inches='tight')
print(f"   Tree structure: {tree_path}")
plt.close()

# 2. Extract and Visualize Decision Rules
print(f"\n Extracting decision rules from tree...")

from sklearn.tree import _tree

def extract_rules(tree, feature_names, class_names):
    """Extract human-readable rules from decision tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, depth, rule_string):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child (<=)
            left_rule = rule_string + f"{name} <= {threshold:.4f}"
            recurse(tree_.children_left[node], depth + 1, left_rule + " AND ")
            
            # Right child (>)
            right_rule = rule_string + f"{name} > {threshold:.4f}"
            recurse(tree_.children_right[node], depth + 1, right_rule + " AND ")
        else:
            # Leaf node
            class_idx = np.argmax(tree_.value[node])
            samples = tree_.n_node_samples[node]
            # Fix: Calculate confidence using weighted values, not sample count
            total_weighted = np.sum(tree_.value[node][0])
            confidence = tree_.value[node][0][class_idx] / total_weighted
            
            rule_string = rule_string.rstrip(" AND ")
            rules.append({
                'rule': rule_string,
                'prediction': class_names[class_idx],
                'samples': samples,
                'confidence': confidence
            })
    
    recurse(0, 0, "IF ")
    return rules

# Extract rules
feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
all_rules = extract_rules(dt_classifier.model, feature_names, CLASS_NAMES)

# Get most confident rule for each class
normal_rules = sorted([r for r in all_rules if r['prediction'] == 'NORMAL'], 
                      key=lambda x: x['confidence'], reverse=True)
pneumonia_rules = sorted([r for r in all_rules if r['prediction'] == 'PNEUMONIA'], 
                         key=lambda x: x['confidence'], reverse=True)

print(f"   Total rules extracted: {len(all_rules)}")
print(f"   NORMAL rules: {len(normal_rules)}")
print(f"   PNEUMONIA rules: {len(pneumonia_rules)}")

# 3. Create NORMAL Rule Visualization (Most Confident Rule - FULL DETAIL)
if normal_rules:
    best_normal_rule = normal_rules[0]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Decision Tree - NORMAL Prediction Rule (Most Confident)', 
                 fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Format the complete rule with all conditions
    conditions = best_normal_rule['rule'].split(' AND ')
    formatted_rule = "COMPLETE RULE FOR PREDICTING NORMAL:\n\n"
    formatted_rule += f"Confidence: {best_normal_rule['confidence']:.1%}\n"
    formatted_rule += f"Training Samples: {best_normal_rule['samples']}\n\n"
    formatted_rule += "-"*80 + "\n\n"
    
    for i, condition in enumerate(conditions, 1):
        formatted_rule += f"Condition {i}:\n  {condition}\n\n"
    
    formatted_rule += "-"*80 + "\n\n"
    formatted_rule += "THEN: PREDICT NORMAL\n\n"
    formatted_rule += f"This rule was learned from {best_normal_rule['samples']} training samples\n"
    formatted_rule += f"and has {best_normal_rule['confidence']:.1%} confidence."
    
    ax.text(0.05, 0.95, formatted_rule, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.5))
    
    plt.tight_layout()
    normal_rule_path = os.path.join(RESULTS_DIR, 'dt_rule_normal.png')
    plt.savefig(normal_rule_path, dpi=150, bbox_inches='tight')
    print(f"   NORMAL rule: {normal_rule_path}")
    plt.close()

# 4. Create PNEUMONIA Rule Visualization (Most Confident Rule - FULL DETAIL)
if pneumonia_rules:
    best_pneumonia_rule = pneumonia_rules[0]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Decision Tree - PNEUMONIA Prediction Rule (Most Confident)', 
                 fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Format the complete rule with all conditions
    conditions = best_pneumonia_rule['rule'].split(' AND ')
    formatted_rule = "COMPLETE RULE FOR PREDICTING PNEUMONIA:\n\n"
    formatted_rule += f"Confidence: {best_pneumonia_rule['confidence']:.1%}\n"
    formatted_rule += f"Training Samples: {best_pneumonia_rule['samples']}\n\n"
    formatted_rule += "-"*80 + "\n\n"
    
    for i, condition in enumerate(conditions, 1):
        formatted_rule += f"Condition {i}:\n  {condition}\n\n"
    
    formatted_rule += "-"*80 + "\n\n"
    formatted_rule += "THEN: PREDICT PNEUMONIA\n\n"
    formatted_rule += f"This rule was learned from {best_pneumonia_rule['samples']} training samples\n"
    formatted_rule += f"and has {best_pneumonia_rule['confidence']:.1%} confidence."
    
    ax.text(0.05, 0.95, formatted_rule, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=1.5))
    
    plt.tight_layout()
    pneumonia_rule_path = os.path.join(RESULTS_DIR, 'dt_rule_pneumonia.png')
    plt.savefig(pneumonia_rule_path, dpi=150, bbox_inches='tight')
    print(f"   PNEUMONIA rule: {pneumonia_rule_path}")
    plt.close()

print(f"\n{'-'*60}")
print("DECISION TREE TRAINING COMPLETE")
print(f"{'-'*60}")
print(f"\nModel saved: {model_path}")
print(f"Results saved: {RESULTS_DIR}/")
