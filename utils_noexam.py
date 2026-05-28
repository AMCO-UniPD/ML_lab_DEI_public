import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt

from utils import stratified_train_test_split, accuracy

class Criterion:
    OPTIMIZATION = ""

    @staticmethod
    def impurity(y) -> float: ...

    @staticmethod
    def split_impurity(y_left, y_right) -> float: ...

    @staticmethod
    def is_tolerance_reached(node_impurity, split_impurity, min_improvement) -> bool: ...

class GiniCriterion(Criterion): 
    OPTIMIZATION = "minimize"

    @staticmethod
    def impurity(y): 
        """Gini impurity is the squared sum of the proportion/probability of each class"""
        class_counts = np.unique(y, return_counts=True)[1]
        probs = class_counts / len(y)
        return 1 - np.sum(probs ** 2)

    @staticmethod 
    def split_impurity(y_left, y_right): 
        """Weighted sum of the gini impurity of two subsets"""
        nL = len(y_left)
        nR = len(y_right)
        n = nL + nR
        g_left = GiniCriterion.impurity(y_left)
        g_right = GiniCriterion.impurity(y_right)
        return (nL * g_left + nR * g_right) / n
    
    @staticmethod 
    def is_tolerance_reached(node_impurity, split_impurity, min_improvement): 
        improvement = node_impurity - split_impurity
        return improvement < min_improvement

@dataclass 
class Node: 
    impurity: float
    split_impurity: float | None    # None if leaf
    feature: int | None             # None if leaf
    threshold: float | None         # None if leaf
    prediction: float | None        # None if not leaf
    n_samples: int

def node_prediction_classification(y): 
    """Return the most common class in the node"""
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

from typing import Type


class DecisionTree: 
    def __init__(self,  max_depth, min_impurity_improvement, criterion: Type[Criterion] = GiniCriterion, node_pred_fnc=node_prediction_classification): 
        """
        Args: 
            criterion: class of the impurity criterion as defined above
            node_pred_fnc: function to predict the class of a node, that takes as input y and return the prediction
            max_depth: maximum depth of the tree
            min_impurity_improvement: minimum impurity improvement to split a node
        """
        # we define here this attribute to share across all the methods 
        self.criterion = criterion
        self.node_pred_fnc = node_pred_fnc
        self.max_depth = max_depth
        self.min_impurity_improvement = min_impurity_improvement

        # attributes to be set in fit method
        self.categorical_feat_idxs = []
        self.nodes = dict()

    @staticmethod
    def _split_dataset(X, feature, threshold, is_categorical): 
        if is_categorical:
            # categorical feature 
            left_mask = X[:, feature] == threshold
            right_mask = X[:, feature] != threshold
        else:
            # numerical feature
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
        return left_mask, right_mask
   
    @staticmethod
    def _child_id(node_id, is_left): 
        if is_left: 
            return 2 * node_id + 1
        else:
            return 2 * node_id + 2
        
    def _find_best_split(self, X, y): 
        """Find the best feature and threshold to split the data"""
        n_features = X.shape[1]

        best_feature:int = -1
        best_threshold:float = 0.0
        best_impurity = np.inf if self.criterion.OPTIMIZATION == "minimize" else -np.inf

        # iterate over features 
        for f in range(n_features):
            is_categorical = f in self.categorical_feat_idxs
            split_impurity, threshold = self._find_best_threshold(X[:, f], y, is_categorical)
            
            if self.criterion.OPTIMIZATION == "minimize":
                is_better = split_impurity < best_impurity  # minimize
            else: 
                is_better = split_impurity > best_impurity  # maximize

            if is_better:
                    best_feature = f
                    best_threshold = threshold
                    best_impurity = split_impurity
        
        return best_impurity, best_feature, best_threshold
    
    def _find_best_threshold(self, X_f, y, is_categorical): 
        """Try all possible thresholds/categories and return the best one"""
        unique_values = np.unique(X_f) 
        best_impurity = np.inf if self.criterion.OPTIMIZATION == "minimize" else -np.inf

        if len(unique_values) < 2:
            return best_impurity, unique_values[0]  # no split possible
        
        for value in unique_values: 
            # divide according to threshold/category
            left_mask, right_mask = self._split_dataset(X_f, ..., value, is_categorical)
            y_left, y_right = y[left_mask], y[right_mask]

            # skip if the value does not split the data
            if len(y_left) == 0 or len(y_right) == 0: 
                continue   

            # evaluate impurity of the split
            split_impurity = self.criterion.split_impurity(y_left, y_right)

            if self.criterion.OPTIMIZATION == "minimize":
                is_better = split_impurity < best_impurity  # minimize
            else:
                is_better = split_impurity > best_impurity  # maximize

            if is_better:
                best_impurity = split_impurity
                best_threshold = value
                    
        return best_impurity, best_threshold

    def _build_tree(self, X, y, depth, node_id): 
        """Construct recursively the tree"""
        n_samples_node = len(y)

        # -- LEAF CONDITIONS ---
        # 0. All the feature values of all the samples are the same (no useful split possible)
        # 1. The node has 0 or 1 samples (no split possible)
        # 2. Max depth is reached
        # 3. The impurity improvement < min_impurity_improvement (implemented after)

        # (leaf condition 0)
        X_all_same = all(len(np.unique(X[:, feat])) == 1 for feat in range(X.shape[1]))

        if len(y) <= 1 or depth >= self.max_depth or X_all_same: 
            prediction = self.node_pred_fnc(y)
            impurity = self.criterion.impurity(y)
            self.nodes[node_id] = Node(impurity, None, None, None, prediction, n_samples_node)    # leaf node
            return

        # -- Internal Node: find best split and create children --
        best_split_impurity, best_feature, best_threshold = self._find_best_split(X, y)

        # (leaf condition 3)
        current_node_impurity = self.criterion.impurity(y)
        if self.criterion.is_tolerance_reached(current_node_impurity, best_split_impurity, self.min_impurity_improvement):
            prediction = self.node_pred_fnc(y)
            impurity = self.criterion.impurity(y)
            self.nodes[node_id] = Node(impurity, None, None, None, prediction, n_samples_node)    # leaf node
            return
        
        # -- Save node --
        self.nodes[node_id] = Node(current_node_impurity, best_split_impurity, best_feature, best_threshold, None, n_samples_node)

        # -- Create children -- 
        is_feat_cat = best_feature in self.categorical_feat_idxs
        left_mask, right_mask = self._split_dataset(X, best_feature, best_threshold, is_feat_cat)

        left_id = self._child_id(node_id, True)
        right_id = self._child_id(node_id, False)

        X_left, y_left = X[left_mask], y[left_mask]
        self._build_tree(X_left, y_left, depth + 1, left_id)

        X_right, y_right = X[right_mask], y[right_mask]
        self._build_tree(X_right, y_right, depth + 1, right_id)

    def fit(self, X, y, categorical_feat_idxs = None): 
        """
        Fit the decision tree to the data. The tree is built recursively by splitting the data at each node
        based on the impurity criterion. 
        Args: 
            X: input matrix of shape (n_samples, n_features)
            y: true target/labels of shape (n_samples,)
            categorical_feat_idxs: list of categorical feature indices
        """
        self.categorical_feat_idxs = categorical_feat_idxs if categorical_feat_idxs is not None else []

        self.nodes = dict()     # Reset the nodes
        self._build_tree(X, y, 0, 0)
        
    def predict(self, X): 
        """
        Predict the target variable for the given input data X. 
        Keep the same column order as in the training data. 
        
        Args: 
            X: input matrix of shape (n_samples, n_features)
        """
        predictions = []
        for x in X: 
            node_id = 0
            node : Node = self.nodes[0]
            while True: 
                if node.prediction is not None: 
                    predictions.append(node.prediction) # leaf node
                    break
                # else we need to go down the tree
                if node.feature in self.categorical_feat_idxs: 
                    # categorical features
                    is_left = x[node.feature] == node.threshold
                else:
                    # continuous features
                    is_left = x[node.feature] <= node.threshold
                node_id = self._child_id(node_id, is_left)
                node = self.nodes[node_id]
        return np.array(predictions)
    

    def print_tree(self, node_id=0, prefx="", is_left=True, feat_names=None):
        """
        Prints the binary tree in a hierarchical format.
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        conn = "├── " if is_left else "└── "

        if node.feature is None:
            if node.impurity is None:
                print(f"{prefx}{conn}{node_id}-Pred: {node.prediction}")
            else:
                print(f"{prefx}{conn}{node_id}-Pred: {node.prediction} | Impurity:{node.impurity:.3f}")
            return
        else:
            if feat_names is None:
                f = f"X[{node.feature}]"
            else:
                f = feat_names[node.feature]
            print(f"{prefx}{conn}{node_id}-{f} <= {node.threshold} | Impurity:{node.impurity:.3f}")

        prefx += "│   " if is_left else "    "
        l_id = self._child_id(node_id, is_left=True)
        r_id = self._child_id(node_id, is_left=False)

        self.print_tree(l_id, prefx, is_left=True, feat_names=feat_names)
        self.print_tree(r_id, prefx, is_left=False, feat_names=feat_names)

def train_ols(X_train, y_train): 
    # Aggiungi una colonna di 1 per il termine di bias (in questo caso sarà il primo coefficiente)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    # Calcola i coefficienti usando la formula OLS: (X^T * X)^-1 * X^T * y
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    return beta

def predict_ols(X_test, beta): 
    y_pred = beta[0] + X_test @ beta[1:]
    return y_pred