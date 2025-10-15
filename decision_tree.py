# Author: Arash Khoeini
# Email: akhoeini@sfu.ca
# Written for SFU CMPT 459

from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
from node import Node

class DecisionTree(object):
    def __init__(self, criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None):
        """
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the tree.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini


    def fit(self, X: pd.DataFrame, y: pd.Series)->float:
        """
        :param X: data
        :param y: label column in X
        :return: accuracy of training dataset
        HINT1: You use self.tree to store the root of the tree
        HINT2: You should use self.split_node to split a node
        """
        # Find the most frequent class for the root node
        node_class = y.mode()[0]
        node_size = len(y)
        single_class = len(y.unique()) == 1
        
        # Create root node
        self.tree = Node(node_size=node_size, node_class=node_class, depth=0, single_class=single_class)
        
        # Split the root node recursively
        self.split_node(self.tree, X, y)
        
        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: predict the class for X.
        HINT1: You can use get_child_node method of Node class to traverse
        HINT2: You can use the mode of the class of the leaf node as the prediction
        HINT3: start traverasl from self.tree
        """
        predictions = []
        
        for _, row in X.iterrows():
            current_node = self.tree
            
            # Traverse the tree until we reach a leaf node
            while not current_node.is_leaf:
                feature_value = row[current_node.name]
                child_node = current_node.get_child_node(feature_value)
                
                # If child node is None (for categorical features with unseen values)
                if child_node is None:
                    break
                current_node = child_node
            
            predictions.append(current_node.node_class)
        
        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> int:
        """
        :param X: data
        :param y: labels
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == y) / len(preds)
        return acc


    def split_node(self, node: Node, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Splits the data in the node into child nodes based on the best feature.

        :param node: the current node to split
        :param X: data in the node
        :param y: labels in the node
        :return: None
        HINT1: Find the best feature to split the data in 'node'.
        HINT2: Use the criterion function (entropy/gini) to score the splits.
        HINT3: Split the data into child nodes
        HINT4: Recursively split the child nodes until the stopping condition is met (e.g., max_depth or single_class).
        """
        # Check stopping conditions
        if self.stopping_condition(node):
            return
        
        # Find the best feature to split on
        best_feature = None
        best_score = float('inf')
        best_threshold = None
        
        for feature in X.columns:
            score = self.criterion_func(X, y, feature)
            if score < best_score:
                best_score = score
                best_feature = feature
                
                # Store the best threshold for numerical features
                if pd.api.types.is_numeric_dtype(X[feature]):
                    best_threshold = self._find_best_threshold(X, y, feature)
        
        # If no good split is found, make it a leaf
        if best_feature is None:
            return
        
        # Set node properties
        node.name = best_feature
        node.is_numerical = pd.api.types.is_numeric_dtype(X[best_feature])
        
        # Create children based on feature type
        children = {}
        
        if node.is_numerical:
            node.threshold = best_threshold
            
            # Split data based on threshold
            left_mask = X[best_feature] < best_threshold
            right_mask = X[best_feature] >= best_threshold
            
            # Create left child (less than threshold)
            if np.any(left_mask):
                left_X = X[left_mask]
                left_y = y[left_mask]
                left_class = left_y.mode()[0] if len(left_y) > 0 else node.node_class
                left_single_class = len(left_y.unique()) == 1 if len(left_y) > 0 else True
                
                left_child = Node(node_size=len(left_y), node_class=left_class, 
                                depth=node.depth + 1, single_class=left_single_class)
                children['l'] = left_child
                
                # Recursively split left child
                self.split_node(left_child, left_X, left_y)
            
            # Create right child (greater than or equal to threshold)
            if np.any(right_mask):
                right_X = X[right_mask]
                right_y = y[right_mask]
                right_class = right_y.mode()[0] if len(right_y) > 0 else node.node_class
                right_single_class = len(right_y.unique()) == 1 if len(right_y) > 0 else True
                
                right_child = Node(node_size=len(right_y), node_class=right_class,
                                 depth=node.depth + 1, single_class=right_single_class)
                children['ge'] = right_child
                
                # Recursively split right child
                self.split_node(right_child, right_X, right_y)
        
        else:
            # Categorical feature
            unique_values = X[best_feature].unique()
            
            for value in unique_values:
                mask = X[best_feature] == value
                child_X = X[mask]
                child_y = y[mask]
                
                if len(child_y) > 0:
                    child_class = child_y.mode()[0]
                    child_single_class = len(child_y.unique()) == 1
                    
                    child = Node(node_size=len(child_y), node_class=child_class,
                               depth=node.depth + 1, single_class=child_single_class)
                    children[value] = child
                    
                    # Recursively split child
                    self.split_node(child, child_X, child_y)
        
        # Set children for the node
        if children:
            node.set_children(children)

    def stopping_condition(self, node: Node) -> bool:
        """
        Checks if the stopping condition for splitting is met.

        :param node: The current node
        :return: True if stopping condition is met, False otherwise
        """
        # Check if the node is pure (all labels are the same)
        if node.single_class:
            return True
        
        # Check if the maximum depth is reached
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        
        # Check if the minimum samples for split is not met
        if self.min_samples_split is not None and node.size < self.min_samples_split:
            return True
        
        return False

    def gini(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns gini index of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get gini score
        :return:
        """
        total_samples = len(y)
        if total_samples == 0:
            return 0
        
        # Check if feature is numerical
        is_numerical = pd.api.types.is_numeric_dtype(X[feature])
        
        if is_numerical:
            # For numerical features, find the best threshold
            unique_values = sorted(X[feature].unique())
            best_gini = float('inf')
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data based on threshold
                left_mask = X[feature] < threshold
                right_mask = X[feature] >= threshold
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Calculate weighted gini
                left_weight = len(left_y) / total_samples
                right_weight = len(right_y) / total_samples
                
                left_gini = self._calculate_gini_impurity(left_y)
                right_gini = self._calculate_gini_impurity(right_y)
                
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
            
            return best_gini if best_gini != float('inf') else 0
        
        else:
            # For categorical features
            weighted_gini = 0
            unique_values = X[feature].unique()
            
            for value in unique_values:
                mask = X[feature] == value
                subset_y = y[mask]
                weight = len(subset_y) / total_samples
                gini_impurity = self._calculate_gini_impurity(subset_y)
                weighted_gini += weight * gini_impurity
            
            return weighted_gini

    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) ->float:
        """
        Returns entropy of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get entropy score
        :return:
        """
        total_samples = len(y)
        if total_samples == 0:
            return 0
        
        # Check if feature is numerical
        is_numerical = pd.api.types.is_numeric_dtype(X[feature])
        
        if is_numerical:
            # For numerical features, find the best threshold
            unique_values = sorted(X[feature].unique())
            best_entropy = float('inf')
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data based on threshold
                left_mask = X[feature] < threshold
                right_mask = X[feature] >= threshold
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Calculate weighted entropy
                left_weight = len(left_y) / total_samples
                right_weight = len(right_y) / total_samples
                
                left_entropy = self._calculate_entropy(left_y)
                right_entropy = self._calculate_entropy(right_y)
                
                weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
                
                if weighted_entropy < best_entropy:
                    best_entropy = weighted_entropy
            
            return best_entropy if best_entropy != float('inf') else 0
        
        else:
            # For categorical features
            weighted_entropy = 0
            unique_values = X[feature].unique()
            
            for value in unique_values:
                mask = X[feature] == value
                subset_y = y[mask]
                weight = len(subset_y) / total_samples
                entropy = self._calculate_entropy(subset_y)
                weighted_entropy += weight * entropy
            
            return weighted_entropy

    def _calculate_gini_impurity(self, y: pd.Series) -> float:
        """
        Calculate Gini impurity for a set of labels
        """
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _calculate_entropy(self, y: pd.Series) -> float:
        """
        Calculate entropy for a set of labels
        """
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
        return entropy

    def _find_best_threshold(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Find the best threshold for a numerical feature
        """
        unique_values = sorted(X[feature].unique())
        best_threshold = None
        best_score = float('inf')
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            # Split data based on threshold
            left_mask = X[feature] < threshold
            right_mask = X[feature] >= threshold
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            # Calculate weighted score
            total_samples = len(y)
            left_weight = len(left_y) / total_samples
            right_weight = len(right_y) / total_samples
            
            if self.criterion == 'entropy':
                left_score = self._calculate_entropy(left_y)
                right_score = self._calculate_entropy(right_y)
            else:
                left_score = self._calculate_gini_impurity(left_y)
                right_score = self._calculate_gini_impurity(right_y)
            
            weighted_score = left_weight * left_score + right_weight * right_score
            
            if weighted_score < best_score:
                best_score = weighted_score
                best_threshold = threshold
        
        return best_threshold if best_threshold is not None else unique_values[0]