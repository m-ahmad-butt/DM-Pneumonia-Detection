import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from helper.tree import (
    print_tree_rules,
    print_model_summary,
    print_confusion_matrix,
    print_metrics
)

print("-" * 80)
print("DECISION TREE TRAINING - STEP BY STEP")
print("-" * 80)

# Paths
FEATURES_DIR = Path(__file__).parent.parent / 'extracted_features'
MODEL_PATH = Path(__file__).parent.parent / 'decision_tree_model.pkl'
MODEL_PATH = Path(__file__).parent.parent / 'decision_tree_model.pkl'

# LOAD DATA
X_train = np.load(FEATURES_DIR / 'X_train_features.npy')
y_train = np.load(FEATURES_DIR / 'y_train.npy')
X_test  = np.load(FEATURES_DIR / 'X_test_features.npy')
y_test  = np.load(FEATURES_DIR / 'y_test.npy')

# MODEL
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# TRAIN
model.fit(X_train, y_train)

# PREDICT
y_pred = model.predict(X_test)

# EVALUATION
cm = confusion_matrix(y_test, y_pred)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
print_model_summary(model)
print_tree_rules(model.tree_, max_depth=5)
print_confusion_matrix(cm)
print_metrics(accuracy, precision, recall, f1)

# SAVE MODEL
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved successfully.")