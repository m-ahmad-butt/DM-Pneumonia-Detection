from sklearn.tree import DecisionTreeClassifier as SKLearnDecisionTree
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10, 
                 random_state=42, use_class_weights=True):
        """
        Initialize Decision Tree classifier
        
        Args:
            max_depth: Maximum depth of tree (prevents overfitting)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required in leaf node
            random_state: Random seed for reproducibility
            use_class_weights: Whether to compute balanced class weights
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.use_class_weights = use_class_weights
        self.class_weight_dict = None
        self.model = None
    
    def fit(self, X, y):
        """
        Train the decision tree on CNN features
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            self
        """
        if self.use_class_weights:
            class_weights_array = compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1]),
                y=y
            )
            self.class_weight_dict = {
                0: class_weights_array[0],  # NORMAL
                1: class_weights_array[1]   # PNEUMONIA
            }
        else:
            self.class_weight_dict = None
        
        self.model = SKLearnDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight_dict,
            random_state=self.random_state
        )
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability matrix (n_samples, 2) for [NORMAL, PNEUMONIA]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_depth(self):
        """Get the depth of the trained tree"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.get_depth()
    
    def get_n_leaves(self):
        """Get the number of leaves in the trained tree"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.get_n_leaves()
    
    @property
    def feature_importances_(self):
        """Get feature importances from the trained tree"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.feature_importances_
    
    @property
    def tree_(self):
        """Get the underlying tree structure (for visualization)"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.tree_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'use_class_weights': self.use_class_weights
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
