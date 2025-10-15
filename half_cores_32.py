import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys

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

def smart_parallel_experiments(X_train, y_train, X_test, y_test, core_fraction=0.5):
    """Smart parallelization with configurable core usage"""
    
    # Calculate cores to use
    total_cores = mp.cpu_count()
    cores_to_use = max(1, int(total_cores * core_fraction))
    
    print(f"\n🧠 SMART PARALLEL MODE:")
    print(f"💻 Total system cores: {total_cores}")
    print(f"🎯 Using cores: {cores_to_use} ({core_fraction*100:.0f}% of system)")
    print(f"🚀 Remaining cores: {total_cores - cores_to_use} (available for other tasks)")
    print("=" * 60)
    
    # Prepare all experiments
    min_sample_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    depths = list(range(1, 17))
    
    all_params = []
    for ms in min_sample_values:
        all_params.append(('min_samples', ms, X_train, y_train, X_test, y_test))
    for depth in depths:
        all_params.append(('max_depth', depth, X_train, y_train, X_test, y_test))
    
    total_experiments = len(all_params)
    print(f"📊 Total experiments: {total_experiments}")
    
    start_time = time.time()
    results = []
    
    # Use specified number of cores
    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        future_to_params = {executor.submit(single_experiment, params): params for params in all_params}
        
        completed = 0
        print(f"\n🔄 Progress (using {cores_to_use} cores):")
        print(f"{'Progress':^10} | {'Type':^12} | {'Param':^6} | {'Train':^7} | {'Test':^7}")
        print("-" * 55)
        
        for future in as_completed(future_to_params):
            result = future.result()
            results.append(result)
            completed += 1
            
            exp_type, param_value, train_acc, test_acc = result
            progress = f"{completed}/{total_experiments}"
            print(f"{progress:^10} | {exp_type:^12} | {param_value:^6} | {train_acc:^7.4f} | {test_acc:^7.4f}")
    
    total_elapsed = time.time() - start_time
    
    # Process results
    min_samples_results = [(r[1], r[2], r[3]) for r in results if r[0] == 'min_samples']
    depth_results = [(r[1], r[2], r[3]) for r in results if r[0] == 'max_depth']
    
    min_samples_results.sort(key=lambda x: x[0])
    depth_results.sort(key=lambda x: x[0])
    
    print(f"\n✅ Completed using {cores_to_use}/{total_cores} cores in {total_elapsed:.1f} seconds")
    
    return min_samples_results, depth_results, total_elapsed

def main():
    """Main function with configurable core usage"""
    
    # Check for command line argument for core fraction
    core_fraction = 0.5  # Default: use half the cores
    
    if len(sys.argv) > 1:
        try:
            core_fraction = float(sys.argv[1])
            core_fraction = max(0.1, min(1.0, core_fraction))  # Clamp between 0.1 and 1.0
        except:
            print("Usage: python half_cores_32.py [core_fraction]")
            print("Example: python half_cores_32.py 0.5  (use 50% of cores)")
            print("Example: python half_cores_32.py 0.25 (use 25% of cores)")
            core_fraction = 0.5
    
    print(f"🚀 SMART CORE ALLOCATION Decision Tree Experiments")
    print(f"🎯 Core usage strategy: {core_fraction*100:.0f}% of available cores")
    print("=" * 60)
    
    total_start = time.time()
    
    # Load data
    X_train, y_train, X_test, y_test = preprocess_data_once()
    
    # Run experiments with specified core fraction
    min_samples_results, depth_results, exec_time = smart_parallel_experiments(
        X_train, y_train, X_test, y_test, core_fraction
    )
    
    # Quick results summary
    ms_test_acc = [r[2] for r in min_samples_results]
    depth_test_acc = [r[2] for r in depth_results]
    
    print(f"\n🏆 RESULTS SUMMARY:")
    print(f"📊 Best min_samples test accuracy: {max(ms_test_acc):.4f}")
    print(f"📊 Best max_depth test accuracy: {max(depth_test_acc):.4f}")
    print(f"⏱️  Execution time: {exec_time:.1f} seconds")
    print(f"💪 Used {int(mp.cpu_count() * core_fraction)} out of {mp.cpu_count()} cores")

if __name__ == '__main__':
    mp.freeze_support()
    main()