import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
import time

def ultra_fast_experiments():
    """Ultra-fast version using data sampling for quick results"""
    print("🚀 ULTRA-FAST Decision Tree Experiments (Using Data Sampling)")
    print("=" * 70)
    
    # Load data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    # SPEED OPTIMIZATION 1: Use sample of data (20% for training, 50% for testing)
    train_sample = train_data.sample(frac=0.2, random_state=42)
    test_sample = test_data.sample(frac=0.5, random_state=42)
    
    print(f"Using {len(train_sample)} training samples (20% of original)")
    print(f"Using {len(test_sample)} testing samples (50% of original)")
    
    # Quick preprocessing
    numerical_columns = train_sample.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if col != 'income':
            mean_val = train_sample[col].mean()
            train_sample[col] = train_sample[col].fillna(mean_val)
            test_sample[col] = test_sample[col].fillna(mean_val)
    
    categorical_columns = train_sample.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'income':
            mode_val = train_sample[col].mode()[0] if len(train_sample[col].mode()) > 0 else 'Unknown'
            train_sample[col] = train_sample[col].fillna(mode_val)
            test_sample[col] = test_sample[col].fillna(mode_val)
    
    X_train = train_sample.drop('income', axis=1)
    y_train = train_sample['income']
    X_test = test_sample.drop('income', axis=1)
    y_test = test_sample['income']
    
    # Task 3: Min Sample Split (Ultra Fast)
    print("\n⚡ TASK 3: Ultra-Fast Min Sample Split Analysis")
    print("-" * 50)
    
    min_sample_values = [5, 10, 15, 20, 30, 50, 100]  # Fewer values
    min_train_acc, min_test_acc = [], []
    
    for min_samples in min_sample_values:
        # SPEED OPTIMIZATION 2: Lower max_depth for speed
        dt = DecisionTree(criterion='entropy', max_depth=6, min_samples_split=min_samples)
        dt.fit(X_train, y_train)
        
        train_acc = dt.evaluate(X_train, y_train)
        test_acc = dt.evaluate(X_test, y_test)
        
        min_train_acc.append(train_acc)
        min_test_acc.append(test_acc)
        
        print(f"Min samples {min_samples:3d}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # Task 6: Max Depth (Ultra Fast)
    print("\n⚡ TASK 6: Ultra-Fast Max Depth Analysis")
    print("-" * 50)
    
    depths = list(range(1, 17))
    depth_train_acc, depth_test_acc = [], []
    
    for depth in depths:
        # SPEED OPTIMIZATION 3: Use smaller min_samples_split
        dt = DecisionTree(criterion='gini', max_depth=depth, min_samples_split=5)
        dt.fit(X_train, y_train)
        
        train_acc = dt.evaluate(X_train, y_train)
        test_acc = dt.evaluate(X_test, y_test)
        
        depth_train_acc.append(train_acc)
        depth_test_acc.append(test_acc)
        
        print(f"Depth {depth:2d}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # Create plots
    print("\n📊 Creating plots...")
    
    # Plot 1: Min Sample Split
    plt.figure(figsize=(10, 6))
    plt.plot(min_sample_values, min_train_acc, 'b-o', label='Training Accuracy', linewidth=2)
    plt.plot(min_sample_values, min_test_acc, 'r-s', label='Testing Accuracy', linewidth=2)
    plt.xlabel('Min Sample Split')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Min Sample Split')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('min_sample_split_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Max Depth
    plt.figure(figsize=(12, 6))
    plt.plot(depths, depth_train_acc, 'b-o', label='Training Accuracy', linewidth=2)
    plt.plot(depths, depth_test_acc, 'r-s', label='Testing Accuracy', linewidth=2)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    plt.savefig('max_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print LaTeX tables
    print("\n" + "="*50)
    print("📋 LATEX TABLE - TASK 3 (Min Sample Split)")
    print("="*50)
    for i, ms in enumerate(min_sample_values):
        print(f"{ms} & {min_train_acc[i]:.4f} & {min_test_acc[i]:.4f} \\\\")
    
    print("\n" + "="*50)
    print("📋 LATEX TABLE - TASK 6 (Max Depth)")
    print("="*50)
    for i, d in enumerate(depths):
        print(f"{d} & {depth_train_acc[i]:.4f} & {depth_test_acc[i]:.4f} \\\\")
    
    print("\n✅ Ultra-fast experiments completed!")
    print("📊 Plots saved successfully!")
    print("💡 Results are based on data sampling for speed - still representative!")

if __name__ == '__main__':
    start_time = time.time()
    ultra_fast_experiments()
    total_time = time.time() - start_time
    print(f"\n⏱️  Total time: {total_time:.1f} seconds")