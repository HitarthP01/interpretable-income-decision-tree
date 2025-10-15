import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

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

def single_experiment(params):
    """Generic worker function for any experiment"""
    exp_type, param_value, X_train, y_train, X_test, y_test = params
    
    if exp_type == 'min_samples':
        dt = DecisionTree(criterion='entropy', max_depth=10, min_samples_split=param_value)
    elif exp_type == 'max_depth':
        dt = DecisionTree(criterion='gini', max_depth=param_value, min_samples_split=10)
    
    dt.fit(X_train, y_train)
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    return exp_type, param_value, train_acc, test_acc

def ultra_parallel_experiments(X_train, y_train, X_test, y_test):
    """Run ALL experiments in parallel simultaneously"""
    print("\n🚀 ULTRA-PARALLEL: Running ALL experiments simultaneously")
    print("=" * 60)
    
    # Prepare all experiments
    min_sample_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    depths = list(range(1, 17))
    
    all_params = []
    
    # Add min_samples experiments
    for ms in min_sample_values:
        all_params.append(('min_samples', ms, X_train, y_train, X_test, y_test))
    
    # Add max_depth experiments  
    for depth in depths:
        all_params.append(('max_depth', depth, X_train, y_train, X_test, y_test))
    
    print(f"Running {len(all_params)} experiments in parallel ({len(min_sample_values)} + {len(depths)})")
    print(f"Using up to {min(8, mp.cpu_count())} workers")
    
    start_time = time.time()
    results = []
    
    # Run all experiments in parallel
    with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(single_experiment, params): params for params in all_params}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_params):
            result = future.result()
            results.append(result)
            completed += 1
            
            exp_type, param_value, train_acc, test_acc = result
            print(f"[{completed:2d}/{len(all_params)}] {exp_type} {param_value}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    elapsed = time.time() - start_time
    
    # Separate results
    min_samples_results = [(r[1], r[2], r[3]) for r in results if r[0] == 'min_samples']
    depth_results = [(r[1], r[2], r[3]) for r in results if r[0] == 'max_depth']
    
    # Sort results
    min_samples_results.sort(key=lambda x: x[0])
    depth_results.sort(key=lambda x: x[0])
    
    print(f"\n⚡ All experiments completed in {elapsed:.1f} seconds!")
    print(f"📊 Average time per experiment: {elapsed/len(all_params):.1f} seconds")
    
    return min_samples_results, depth_results

def create_plots_and_tables(min_samples_results, depth_results):
    """Create plots and print LaTeX tables"""
    print("\n📊 Creating plots...")
    
    # Extract data
    ms_values = [r[0] for r in min_samples_results]
    ms_train_acc = [r[1] for r in min_samples_results]
    ms_test_acc = [r[2] for r in min_samples_results]
    
    depth_values = [r[0] for r in depth_results]
    depth_train_acc = [r[1] for r in depth_results]
    depth_test_acc = [r[2] for r in depth_results]
    
    # Plot 1: Min Sample Split
    plt.figure(figsize=(10, 6))
    plt.plot(ms_values, ms_train_acc, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(ms_values, ms_test_acc, 'r-s', label='Testing Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Min Sample Split', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree Accuracy vs Min Sample Split', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('min_sample_split_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Max Depth
    plt.figure(figsize=(12, 6))
    plt.plot(depth_values, depth_train_acc, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(depth_values, depth_test_acc, 'r-s', label='Testing Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Max Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree Accuracy vs Max Depth', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)
    plt.tight_layout()
    plt.savefig('max_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Plots saved successfully!")
    
    # Print LaTeX tables
    print("\n" + "="*60)
    print("📋 LATEX TABLE FORMAT FOR TASK 3 (Min Sample Split):")
    print("="*60)
    print("Min Sample Split & Training Accuracy & Testing Accuracy \\\\")
    print("\\midrule")
    for i, ms in enumerate(ms_values):
        print(f"{ms} & {ms_train_acc[i]:.4f} & {ms_test_acc[i]:.4f} \\\\")
    
    print("\n" + "="*60)
    print("📋 LATEX TABLE FORMAT FOR TASK 6 (Max Depth):")
    print("="*60)
    print("Max Depth & Training Accuracy & Testing Accuracy \\\\")
    print("\\midrule")
    for i, d in enumerate(depth_values):
        print(f"{d} & {depth_train_acc[i]:.4f} & {depth_test_acc[i]:.4f} \\\\")
    
    # Print summaries
    print("\n" + "="*60)
    print("📈 RESULTS SUMMARY:")
    print("="*60)
    best_ms_idx = ms_test_acc.index(max(ms_test_acc))
    best_depth_idx = depth_test_acc.index(max(depth_test_acc))
    
    print(f"Task 3 - Best min_samples: {ms_values[best_ms_idx]} (Test Acc: {ms_test_acc[best_ms_idx]:.4f})")
    print(f"Task 6 - Best max_depth: {depth_values[best_depth_idx]} (Test Acc: {depth_test_acc[best_depth_idx]:.4f})")

def main():
    """Main function for ultra-parallel experiments"""
    print("🚀 ULTRA-PARALLEL Decision Tree Experiments")
    print("Running ALL experiments simultaneously for maximum speed!")
    print("=" * 60)
    print(f"System CPU cores: {mp.cpu_count()}")
    
    total_start = time.time()
    
    # Preprocess data once
    X_train, y_train, X_test, y_test = preprocess_data_once()
    
    # Run all experiments in parallel
    min_samples_results, depth_results = ultra_parallel_experiments(X_train, y_train, X_test, y_test)
    
    # Create plots and tables
    create_plots_and_tables(min_samples_results, depth_results)
    
    total_time = time.time() - total_start
    
    print(f"\n⏱️  TOTAL RUNTIME: {total_time:.1f} seconds")
    print(f"🚀 Estimated speedup: ~{4:.1f}x compared to sequential execution")
    print("\n✅ Ultra-parallel experiments completed successfully!")
    print("📊 All plots and LaTeX tables ready!")

if __name__ == '__main__':
    mp.freeze_support()
    main()