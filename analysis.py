import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import argparse

def read_and_preprocess_data():
    """Load and preprocess the data"""
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    # Handle missing values
    numerical_columns = train_data.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if col != 'income':
            mean_val = train_data[col].mean()
            train_data[col] = train_data[col].fillna(mean_val)
            test_data[col] = test_data[col].fillna(mean_val)
    
    categorical_columns = train_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'income':
            mode_val = train_data[col].mode()[0] if len(train_data[col].mode()) > 0 else 'Unknown'
            train_data[col] = train_data[col].fillna(mode_val)
            test_data[col] = test_data[col].fillna(mode_val)
    
    X_train = train_data.drop('income', axis=1)
    y_train = train_data['income']
    X_test = test_data.drop('income', axis=1)
    y_test = test_data['income']
    
    return X_train, y_train, X_test, y_test

def task1_gini():
    """Task 1: Train with Gini index"""
    print("=== TASK 1: Gini Index ===")
    X_train, y_train, X_test, y_test = read_and_preprocess_data()
    
    dt = DecisionTree(criterion='gini', max_depth=10, min_samples_split=20)
    dt.fit(X_train, y_train)
    
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print()
    
    return train_acc, test_acc

def task2_entropy():
    """Task 2: Train with Information Gain (Entropy)"""
    print("=== TASK 2: Information Gain (Entropy) ===")
    X_train, y_train, X_test, y_test = read_and_preprocess_data()
    
    dt = DecisionTree(criterion='entropy', max_depth=10, min_samples_split=20)
    dt.fit(X_train, y_train)
    
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print()
    
    return train_acc, test_acc

def task3_min_samples_analysis():
    """Task 3: Analyze different min_sample_split values"""
    print("=== TASK 3: Min Sample Split Analysis ===")
    X_train, y_train, X_test, y_test = read_and_preprocess_data()
    
    min_sample_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    train_accuracies = []
    test_accuracies = []
    
    for min_samples in min_sample_values:
        dt = DecisionTree(criterion='entropy', max_depth=10, min_samples_split=min_samples)
        dt.fit(X_train, y_train)
        
        train_acc = dt.evaluate(X_train, y_train)
        test_acc = dt.evaluate(X_test, y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Min samples: {min_samples:3d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(min_sample_values, train_accuracies, 'b-o', label='Training Accuracy')
    plt.plot(min_sample_values, test_accuracies, 'r-s', label='Testing Accuracy')
    plt.xlabel('Min Sample Split')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Min Sample Split')
    plt.legend()
    plt.grid(True)
    plt.savefig('min_sample_split_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print()
    
    return min_sample_values, train_accuracies, test_accuracies

def task6_depth_analysis():
    """Task 6: Analyze different max_depth values"""
    print("=== TASK 6: Max Depth Analysis ===")
    X_train, y_train, X_test, y_test = read_and_preprocess_data()
    
    depths = list(range(1, 17))
    train_accuracies = []
    test_accuracies = []
    
    for depth in depths:
        dt = DecisionTree(criterion='gini', max_depth=depth, min_samples_split=10)
        dt.fit(X_train, y_train)
        
        train_acc = dt.evaluate(X_train, y_train)
        test_acc = dt.evaluate(X_test, y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Max depth: {depth:2d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(depths, train_accuracies, 'b-o', label='Training Accuracy')
    plt.plot(depths, test_accuracies, 'r-s', label='Testing Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    plt.savefig('max_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print()
    
    return depths, train_accuracies, test_accuracies

def perfect_training_accuracy():
    """Test for perfect training accuracy"""
    print("=== Testing for Perfect Training Accuracy ===")
    X_train, y_train, X_test, y_test = read_and_preprocess_data()
    
    # Try with very deep tree and small min_samples_split
    dt = DecisionTree(criterion='gini', max_depth=None, min_samples_split=2)
    dt.fit(X_train, y_train)
    
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    print(f"Deep tree (no max depth, min_samples=2):")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print()
    
    return train_acc, test_acc

def main():
    print("Decision Tree Analysis Report")
    print("=" * 50)
    print()
    
    # Task 1
    gini_train, gini_test = task1_gini()
    
    # Task 2
    entropy_train, entropy_test = task2_entropy()
    
    # Task 3
    min_samples, train_accs, test_accs = task3_min_samples_analysis()
    
    # Task 6
    depths, depth_train_accs, depth_test_accs = task6_depth_analysis()
    
    # Perfect training accuracy test
    perfect_train, perfect_test = perfect_training_accuracy()
    
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Task 1 (Gini): Train={gini_train:.4f}, Test={gini_test:.4f}")
    print(f"Task 2 (Entropy): Train={entropy_train:.4f}, Test={entropy_test:.4f}")
    print(f"Best min_samples performance: {max(test_accs):.4f} at min_samples={min_samples[test_accs.index(max(test_accs))]}")
    print(f"Best depth performance: {max(depth_test_accs):.4f} at depth={depths[depth_test_accs.index(max(depth_test_accs))]}")
    print(f"Perfect training setup: Train={perfect_train:.4f}, Test={perfect_test:.4f}")

if __name__ == '__main__':
    main()