# Interpretable Income Decision Tree

A from-scratch implementation of a CART-style decision tree that predicts whether an individual earns more than $50K using the UCI Adult Income dataset. The project combines clean, explainable machine learning with reproducible experimentation and high-performance tooling—no scikit-learn required.

## Highlights

- **Hand-built learner** – Implemented `fit`, `split_node`, `predict`, `gini`, and `entropy` entirely from scratch with just NumPy and pandas.
- **Interpretability-first** – Uses the `Node` class to surface thresholds, categorical splits, and leaf statistics for easy inspection and reporting.
- **Robust preprocessing** – Handles numerical and categorical missing values consistently across train and test sets.
- **Experiment automation** – Includes parallelized scripts that sweep hyperparameters across 10+ `min_samples_split` values and 16 tree depths, generating publication-ready plots.
- **Scales to serious hardware** – Optimized runners (`max_cores_32.py`, `half_cores_32.py`) saturate multi-core workstations while preserving reproducible outputs.
- **Comprehensive report** – `report.md` / `report.tex` document methodology, pseudo-code, results, and discussion suitable for academic submission.

## Quick Start

```bash
# Activate the venv (optional, but recommended)
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt  # or manually install numpy pandas matplotlib

# Train and evaluate the baseline decision tree
python main.py --train-data data/train.csv --test-data data/test.csv \
               --criterion gini --maxdepth 10 --min-sample-split 20
```

Example output:
```
Training Accuracy: 0.8343
Testing Accuracy: 0.8305
```

## Repository Structure

```
├── decision_tree.py        # Core DecisionTree class (from scratch)
├── node.py                 # Node data structure used during training/inference
├── main.py                 # CLI entry point for training/evaluation
├── data/                   # Adult Income train/test CSV files
├── analysis.py             # End-to-end experiment runner (baseline version)
├── fast_experiments.py     # Parallelized min_samples_split & max_depth sweeps
├── parallel_experiments.py # Improved parallel runner for multi-core systems
├── ultra_parallel_experiments.py
├── max_cores_32.py         # 32-core optimized experiment runner
├── half_cores_32.py        # Fractional core runner for concurrent workloads
├── min_sample_split_analysis.png
├── max_depth_analysis.png
├── report.md / report.tex  # Full assignment report (results, discussion)
└── README.md               # You are here
```

## Reproducing the Experiments

### 1. Criterion Comparison

```bash
python main.py --criterion gini --maxdepth 10 --min-sample-split 20
python main.py --criterion entropy --maxdepth 10 --min-sample-split 20
```

| Criterion | Train Accuracy | Test Accuracy |
|-----------|----------------|---------------|
| Gini      | 0.8343         | 0.8305        |
| Entropy   | 0.8341         | 0.8300        |

### 2. Min Sample Split Sweep

```bash
python max_cores_32.py  # Uses all available CPU cores to sweep
```

Result summary:
- Best `min_samples_split`: **25** (test accuracy **0.8302**)
- High-performing band: 25–40
- Generates `min_sample_split_analysis.png`

### 3. Max Depth Sweep

```bash
python max_cores_32.py  # Same run produces both plots
```

Result summary:
- Best `max_depth`: **3** (test accuracy **0.8308**)
- Shallow trees generalize best; deeper trees plateau
- Generates `max_depth_analysis.png`

### 4. Scaled Parallel Experiments

If multiple experiments must run simultaneously, fractionally allocate cores:

```bash
# Use 50% of the machine (e.g., 16 of 32 cores)
python half_cores_32.py 0.5

# Lightweight sampling sweeps (fast prototyping)
python ultra_fast_experiments.py
```

## Dataset

The project ships with pre-split train/test CSVs derived from the [UCI Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult). Each record contains 14 demographic and employment features plus the binary target `income` (`<=50K`, `>50K`).

## Reporting & Documentation

- `report.md` provides a readable Markdown summary of the assignment deliverables.
- `report.tex` compiles to a polished PDF with pseudo-code, discussions on bias/variance, and experimental results.
- Plots are generated at 300 DPI for inclusion in academic or professional presentations.

## Future Ideas

- Implement pruning strategies (cost-complexity, reduced-error) for deeper interpretability.
- Add cross-validation for more robust hyperparameter tuning.
- Extend the CLI with model export (e.g., JSON representation of the trained tree).

## License

This repository is intended for educational and portfolio use. If you plan to reuse the code in coursework, please follow your institution’s academic integrity guidelines.
