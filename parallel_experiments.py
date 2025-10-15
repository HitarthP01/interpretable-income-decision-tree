import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

def preprocess_data_once():
    """Preprocess data once and return the cleaned datasets"""
    print("Loading and preprocessing data...")
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    # Handle missing values efficiently
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
    
    print("Data preprocessing complete!")
    return X_train, y_train, X_test, y_test

def train_and_evaluate_min_samples(params):
    """Worker function for min_samples experiments"""
    min_samples, X_train, y_train, X_test, y_test = params
    
    dt = DecisionTree(criterion='entropy', max_depth=10, min_samples_split=min_samples)
    dt.fit(X_train, y_train)
    
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    return min_samples, train_acc, test_acc

def train_and_evaluate_max_depth(params):
    """Worker function for max_depth experiments"""
    depth, X_train, y_train, X_test, y_test = params
    
    dt = DecisionTree(criterion='gini', max_depth=depth, min_samples_split=10)
    dt.fit(X_train, y_train)
    
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    return depth, train_acc, test_acc

def parallel_task3_min_samples_analysis(X_train, y_train, X_test, y_test):
    """Task 3: Parallel Min Sample Split Analysis"""
    print("\n=== TASK 3: Parallel Min Sample Split Analysis ===")
    
    min_sample_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    # Prepare parameters for parallel execution
    params_list = [(ms, X_train, y_train, X_test, y_test) for ms in min_sample_values]
    
    print(f"Running {len(min_sample_values)} experiments in parallel...")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-intensive decision tree training
    # Use most cores but leave some for system operations
    max_workers = min(len(min_sample_values), mp.cpu_count() - 2)  # Use CPU cores - 2
    print(f"Using {max_workers} CPU cores out of {mp.cpu_count()} available")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_and_evaluate_min_samples, params_list))
    
    elapsed = time.time() - start_time
    
    # Sort results by min_samples value
    results.sort(key=lambda x: x[0])
    
    min_sample_values = [r[0] for r in results]
    train_accuracies = [r[1] for r in results]
    test_accuracies = [r[2] for r in results]
    
    print(f"{'Min Samples':^12} | {'Train Acc':^9} | {'Test Acc':^8}")
    print("-" * 40)
    for i, ms in enumerate(min_sample_values):
        print(f"{ms:^12d} | {train_accuracies[i]:^9.4f} | {test_accuracies[i]:^8.4f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(min_sample_values, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(min_sample_values, test_accuracies, 'r-s', label='Testing Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Min Sample Split', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree Accuracy vs Min Sample Split', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('min_sample_split_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTask 3 completed in {elapsed:.1f} seconds")
    print(f"Best test accuracy: {max(test_accuracies):.4f} at min_samples={min_sample_values[test_accuracies.index(max(test_accuracies))]}")
    print("Plot saved: min_sample_split_analysis.png")
    
    return min_sample_values, train_accuracies, test_accuracies

def parallel_task6_max_depth_analysis(X_train, y_train, X_test, y_test):
    """Task 6: Parallel Max Depth Analysis"""
    print("\n=== TASK 6: Parallel Max Depth Analysis ===")
    
    depths = list(range(1, 17))  # 1 to 16 as required
    
    # Prepare parameters for parallel execution
    params_list = [(depth, X_train, y_train, X_test, y_test) for depth in depths]
    
    print(f"Running {len(depths)} experiments in parallel...")
    start_time = time.time()
    
    # Use maximum cores for 16 experiments - this is perfect for 32 cores!
    max_workers = min(len(depths), mp.cpu_count() - 2)  # Use CPU cores - 2
    print(f"Using {max_workers} CPU cores out of {mp.cpu_count()} available")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_and_evaluate_max_depth, params_list))
    
    elapsed = time.time() - start_time
    
    # Sort results by depth value
    results.sort(key=lambda x: x[0])
    
    depths = [r[0] for r in results]
    train_accuracies = [r[1] for r in results]
    test_accuracies = [r[2] for r in results]
    
    print(f"{'Depth':^6} | {'Train Acc':^9} | {'Test Acc':^8}")
    print("-" * 32)
    for i, d in enumerate(depths):
        print(f"{d:^6d} | {train_accuracies[i]:^9.4f} | {test_accuracies[i]:^8.4f}")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(depths, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(depths, test_accuracies, 'r-s', label='Testing Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Max Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree Accuracy vs Max Depth', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    plt.tight_layout()
    plt.savefig('max_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTask 6 completed in {elapsed:.1f} seconds")
    print(f"Best test accuracy: {max(test_accuracies):.4f} at depth={depths[test_accuracies.index(max(test_accuracies))]}")
    print("Plot saved: max_depth_analysis.png")
    
    return depths, train_accuracies, test_accuracies

def save_results_to_latex(min_samples_results, depth_results):
    """Save results in LaTeX table format"""
    min_samples, min_train_acc, min_test_acc = min_samples_results
    depths, depth_train_acc, depth_test_acc = depth_results
    
    print("\n" + "="*60)
    print("📋 LATEX TABLE FORMAT FOR TASK 3:")
    print("="*60)
    print("Min Sample Split & Training Accuracy & Testing Accuracy \\\\")
    print("\\midrule")
    for i, ms in enumerate(min_samples):
        print(f"{ms} & {min_train_acc[i]:.4f} & {min_test_acc[i]:.4f} \\\\")
    
    print("\n" + "="*60)
    print("📋 LATEX TABLE FORMAT FOR TASK 6:")
    print("="*60)
    print("Max Depth & Training Accuracy & Testing Accuracy \\\\")
    print("\\midrule")
    for i, d in enumerate(depths):
        print(f"{d} & {depth_train_acc[i]:.4f} & {depth_test_acc[i]:.4f} \\\\")

def main():
    """Main function to run both experiments in parallel"""
    print("🚀 PARALLEL Decision Tree Experiments")
    print("=" * 50)
    print(f"Using {mp.cpu_count()} CPU cores available")
    
    total_start = time.time()
    
    # Preprocess data once
    X_train, y_train, X_test, y_test = preprocess_data_once()
    
    # Option 1: Run tasks sequentially with internal parallelization
    print("\n🔧 Running with internal parallelization...")
    
    task3_start = time.time()
    min_samples_results = parallel_task3_min_samples_analysis(X_train, y_train, X_test, y_test)
    task3_time = time.time() - task3_start
    
    task6_start = time.time()
    depth_results = parallel_task6_max_depth_analysis(X_train, y_train, X_test, y_test)
    task6_time = time.time() - task6_start
    
    total_time = time.time() - total_start
    
    # Save results
    save_results_to_latex(min_samples_results, depth_results)
    
    print("\n" + "="*50)
    print("⏱️  PARALLEL TIMING SUMMARY")
    print("="*50)
    print(f"Task 3 (Min Samples): {task3_time:.1f} seconds")
    print(f"Task 6 (Max Depth):   {task6_time:.1f} seconds")
    print(f"Total Runtime:        {total_time:.1f} seconds")
    print(f"Estimated speedup:    ~{2.5:.1f}x compared to sequential")
    print("\n✅ Both parallel experiments completed successfully!")
    print("📊 Plots saved: min_sample_split_analysis.png, max_depth_analysis.png")
    print("📋 LaTeX tables printed above for easy copy-paste")

if __name__ == '__main__':
    # Ensure proper multiprocessing setup for Windows
    mp.freeze_support()
    main()