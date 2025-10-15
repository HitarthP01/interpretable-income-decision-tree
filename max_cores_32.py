import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    """Generic worker function for any experiment - optimized for 32 cores"""
    exp_type, param_value, X_train, y_train, X_test, y_test = params
    
    if exp_type == 'min_samples':
        dt = DecisionTree(criterion='entropy', max_depth=10, min_samples_split=param_value)
    elif exp_type == 'max_depth':
        dt = DecisionTree(criterion='gini', max_depth=param_value, min_samples_split=10)
    
    dt.fit(X_train, y_train)
    train_acc = dt.evaluate(X_train, y_train)
    test_acc = dt.evaluate(X_test, y_test)
    
    return exp_type, param_value, train_acc, test_acc

def max_cores_experiments(X_train, y_train, X_test, y_test):
    """Use ALL 32 cores simultaneously for maximum speed"""
    print(f"\n🚀 MAX CORES MODE: Using all {mp.cpu_count()} CPU cores simultaneously!")
    print("=" * 70)
    
    # Prepare all experiments
    min_sample_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    depths = list(range(1, 17))  # 1-16 as required
    
    all_params = []
    
    # Add min_samples experiments (Task 3)
    for ms in min_sample_values:
        all_params.append(('min_samples', ms, X_train, y_train, X_test, y_test))
    
    # Add max_depth experiments (Task 6)
    for depth in depths:
        all_params.append(('max_depth', depth, X_train, y_train, X_test, y_test))
    
    total_experiments = len(all_params)
    print(f"📊 Total experiments: {total_experiments} ({len(min_sample_values)} min_samples + {len(depths)} max_depth)")
    print(f"💪 Using {mp.cpu_count()} CPU cores (all available)")
    print(f"⚡ Expected massive speedup with 32-core parallelization!")
    
    start_time = time.time()
    results = []
    
    # Use ALL CPU cores for maximum performance
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Submit all jobs at once
        future_to_params = {executor.submit(single_experiment, params): params for params in all_params}
        
        # Collect results with progress tracking
        completed = 0
        print(f"\n🔄 Progress tracking:")
        print(f"{'Progress':^10} | {'Type':^12} | {'Param':^6} | {'Train':^7} | {'Test':^7} | {'Time':^6}")
        print("-" * 65)
        
        for future in as_completed(future_to_params):
            result = future.result()
            results.append(result)
            completed += 1
            
            exp_type, param_value, train_acc, test_acc = result
            elapsed = time.time() - start_time
            progress = f"{completed}/{total_experiments}"
            
            print(f"{progress:^10} | {exp_type:^12} | {param_value:^6} | {train_acc:^7.4f} | {test_acc:^7.4f} | {elapsed:^6.1f}s")
    
    total_elapsed = time.time() - start_time
    
    # Separate and sort results
    min_samples_results = [(r[1], r[2], r[3]) for r in results if r[0] == 'min_samples']
    depth_results = [(r[1], r[2], r[3]) for r in results if r[0] == 'max_depth']
    
    min_samples_results.sort(key=lambda x: x[0])
    depth_results.sort(key=lambda x: x[0])
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"⏱️  Total time with 32 cores: {total_elapsed:.1f} seconds")
    print(f"📈 Average time per experiment: {total_elapsed/total_experiments:.2f} seconds")
    print(f"🚀 Estimated speedup vs sequential: ~{min(32, total_experiments):.0f}x")
    
    return min_samples_results, depth_results, total_elapsed

def create_plots_and_results(min_samples_results, depth_results, execution_time):
    """Create plots and display all results"""
    print("\n📊 Creating visualization plots...")
    
    # Extract data for plotting
    ms_values = [r[0] for r in min_samples_results]
    ms_train_acc = [r[1] for r in min_samples_results]
    ms_test_acc = [r[2] for r in min_samples_results]
    
    depth_values = [r[0] for r in depth_results]
    depth_train_acc = [r[1] for r in depth_results]
    depth_test_acc = [r[2] for r in depth_results]
    
    # Create high-quality plots
    plt.style.use('default')
    
    # Plot 1: Min Sample Split Analysis
    plt.figure(figsize=(12, 7))
    plt.plot(ms_values, ms_train_acc, 'b-o', label='Training Accuracy', linewidth=3, markersize=8)
    plt.plot(ms_values, ms_test_acc, 'r-s', label='Testing Accuracy', linewidth=3, markersize=8)
    plt.xlabel('Min Sample Split', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Decision Tree Accuracy vs Min Sample Split\n(32-Core Parallel Execution)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('min_sample_split_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Max Depth Analysis
    plt.figure(figsize=(14, 7))
    plt.plot(depth_values, depth_train_acc, 'b-o', label='Training Accuracy', linewidth=3, markersize=8)
    plt.plot(depth_values, depth_test_acc, 'r-s', label='Testing Accuracy', linewidth=3, markersize=8)
    plt.xlabel('Max Depth', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Decision Tree Accuracy vs Max Depth\n(32-Core Parallel Execution)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)
    plt.tight_layout()
    plt.savefig('max_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ High-quality plots saved!")
    
    # Display results summary
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*70)
    
    best_ms_idx = ms_test_acc.index(max(ms_test_acc))
    best_depth_idx = depth_test_acc.index(max(depth_test_acc))
    
    print(f"📊 Task 3 Results:")
    print(f"   • Best min_samples_split: {ms_values[best_ms_idx]}")
    print(f"   • Best test accuracy: {ms_test_acc[best_ms_idx]:.4f}")
    print(f"   • Training accuracy: {ms_train_acc[best_ms_idx]:.4f}")
    
    print(f"\n📊 Task 6 Results:")
    print(f"   • Best max_depth: {depth_values[best_depth_idx]}")
    print(f"   • Best test accuracy: {depth_test_acc[best_depth_idx]:.4f}")
    print(f"   • Training accuracy: {depth_train_acc[best_depth_idx]:.4f}")
    
    print(f"\n⚡ Performance:")
    print(f"   • Total execution time: {execution_time:.1f} seconds")
    print(f"   • Used all {mp.cpu_count()} CPU cores")
    print(f"   • Parallel efficiency: MAXIMUM")
    
    # Print LaTeX tables
    print("\n" + "="*70)
    print("📋 LATEX TABLE - TASK 3 (Min Sample Split)")
    print("="*70)
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("Min Sample Split & Training Accuracy & Testing Accuracy \\\\")
    print("\\midrule")
    for i, ms in enumerate(ms_values):
        print(f"{ms} & {ms_train_acc[i]:.4f} & {ms_test_acc[i]:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    print("\n" + "="*70)
    print("📋 LATEX TABLE - TASK 6 (Max Depth)")
    print("="*70)
    print("\\begin{tabular}{ccc}")
    print("\\toprule")
    print("Max Depth & Training Accuracy & Testing Accuracy \\\\")
    print("\\midrule")
    for i, d in enumerate(depth_values):
        print(f"{d} & {depth_train_acc[i]:.4f} & {depth_test_acc[i]:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

def main():
    """Main function optimized for 32-core execution"""
    print("🚀 32-CORE MAXIMUM PERFORMANCE Decision Tree Experiments")
    print("🔥 UNLEASHING THE FULL POWER OF YOUR SFU WORKSTATION!")
    print("=" * 70)
    
    if mp.cpu_count() < 16:
        print(f"⚠️  Warning: Only {mp.cpu_count()} cores detected. This script is optimized for 32+ cores.")
    else:
        print(f"💪 Perfect! {mp.cpu_count()} cores detected - ready for maximum performance!")
    
    total_start = time.time()
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = preprocess_data_once()
    
    # Run all experiments with maximum parallelization
    min_samples_results, depth_results, exec_time = max_cores_experiments(X_train, y_train, X_test, y_test)
    
    # Generate final plots and results
    create_plots_and_results(min_samples_results, depth_results, exec_time)
    
    total_time = time.time() - total_start
    
    print(f"\n🎯 MISSION ACCOMPLISHED!")
    print(f"⏱️  Total runtime: {total_time:.1f} seconds")
    print(f"🚀 Your 32-core beast delivered maximum performance!")
    print(f"📊 All plots and LaTeX tables are ready for your report!")

if __name__ == '__main__':
    # Ensure proper multiprocessing on all platforms
    mp.freeze_support()
    main()