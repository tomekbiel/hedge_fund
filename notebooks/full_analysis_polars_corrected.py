# FULL DATASET ANALYSIS (POLARS)
# ============================================
# IMPORTS AND CONFIGURATION
# ============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import polars as pl
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
import time
from collections import Counter

# Polars configuration
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

# Set up paths
base_dir = Path("..")
full_train_path = base_dir / "data" / "train.parquet"
full_test_path = base_dir / "data" / "test.parquet"

print("✅ Imports and configuration complete")

# ============================================
# LOAD FULL DATASETS
# ============================================
print("="*60)
print("LOADING FULL DATASETS")
print("="*60)

# Load full datasets using Polars
start_time = time.time()
train_full = pl.read_parquet(full_train_path)
test_full = pl.read_parquet(full_test_path)
load_time = time.time() - start_time

print(f"Train full: {train_full.shape}")
print(f"Test full: {test_full.shape}")
print(f"Load time: {load_time:.2f} seconds")

print(f"\n✅ Full datasets loaded")
print(f"   Train: {train_full.shape}")
print(f"   Test: {test_full.shape}")

# ============================================
# BASIC DATA STRUCTURE ANALYSIS
# ============================================
print("="*60)
print("DATA STRUCTURE ANALYSIS")
print("="*60)

# Train dataset structure
print("\n📋 TRAIN DATASET STRUCTURE:")
print(f"Shape: {train_full.shape}")
print(f"Columns: {train_full.columns}")
print(f"\nData types:")
# Correct way to get dtype counts in Polars
train_dtype_counts = Counter(str(dtype) for dtype in train_full.dtypes)
for dtype, count in train_dtype_counts.items():
    print(f"{dtype}: {count}")

# Test dataset structure  
print("\n📋 TEST DATASET STRUCTURE:")
print(f"Shape: {test_full.shape}")
print(f"Columns: {test_full.columns}")
print(f"\nData types:")
# Correct way to get dtype counts in Polars
test_dtype_counts = Counter(str(dtype) for dtype in test_full.dtypes)
for dtype, count in test_dtype_counts.items():
    print(f"{dtype}: {count}")

# Column differences
train_cols = set(train_full.columns)
test_cols = set(test_full.columns)
only_in_train = train_cols - test_cols
only_in_test = test_cols - train_cols

print(f"\n📊 COLUMN DIFFERENCES:")
print(f"Only in train: {only_in_train}")
print(f"Only in test: {only_in_test}")
print(f"Common columns: {len(train_cols & test_cols)}")

# ============================================
# DETAILED COLUMN INFORMATION
# ============================================
print("="*60)
print("DETAILED COLUMN ANALYSIS")
print("="*60)

# Categorical columns analysis
cat_cols = ['code', 'sub_code', 'sub_category']
print("\n🏷️ CATEGORICAL COLUMNS:")
for col in cat_cols:
    if col in train_full.columns:
        unique_count = train_full[col].n_unique()
        print(f"\n{col}:")
        print(f"  Unique values: {unique_count}")
        print(f"  Sample values: {train_full[col].unique().head(10).to_list()}")
        if col == 'sub_category':
            print(f"  Value counts:")
            value_counts_df = train_full[col].value_counts()
            print(value_counts_df)

# Temporal columns
temporal_cols = ['ts_index', 'horizon']
print("\n⏰ TEMPORAL COLUMNS:")
for col in temporal_cols:
    if col in train_full.columns:
        print(f"\n{col}:")
        print(f"  Min: {train_full[col].min()}")
        print(f"  Max: {train_full[col].max()}")
        print(f"  Unique values: {train_full[col].n_unique()}")

# Target and weight
special_cols = ['y_target', 'weight']
print("\n🎯 SPECIAL COLUMNS:")
for col in special_cols:
    if col in train_full.columns:
        print(f"\n{col}:")
        print(f"  Type: {train_full[col].dtype}")
        print(f"  Min: {train_full[col].min()}")
        print(f"  Max: {train_full[col].max()}")
        print(f"  Mean: {train_full[col].mean():.6f}")
        print(f"  Std: {train_full[col].std():.6f}")

# ============================================
# MISSING VALUES ANALYSIS
# ============================================
print("="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)

# Missing values in train dataset
print("\n🔍 TRAIN DATASET MISSING VALUES:")
train_nulls = train_full.null_count()
train_null_cols = train_nulls.filter(pl.col(train_nulls.columns[0]) > 0)

if len(train_null_cols) > 0:
    print(train_null_cols)
else:
    print("✅ No missing values in train dataset")

# Missing values in test dataset
print("\n🔍 TEST DATASET MISSING VALUES:")
test_nulls = test_full.null_count()
test_null_cols = test_nulls.filter(pl.col(test_nulls.columns[0]) > 0)

if len(test_null_cols) > 0:
    print(test_null_cols)
else:
    print("✅ No missing values in test dataset")

# Missing value percentages
print("\n📊 MISSING VALUE PERCENTAGES:")
train_total = len(train_full)
test_total = len(test_full)

print(f"Train dataset total rows: {train_total:,}")
print(f"Test dataset total rows: {test_total:,}")

# ============================================
# DISTRIBUTION ANALYSIS
# ============================================
print("="*60)
print("DISTRIBUTION ANALYSIS")
print("="*60)

# Get numeric columns using Polars
numeric_cols = []
for col in train_full.columns:
    dtype = train_full[col].dtype
    if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
        numeric_cols.append(col)

print(f"\n📈 Numeric columns found: {len(numeric_cols)}")
print(f"First 10 numeric columns: {numeric_cols[:10]}")

# Basic statistics for numeric columns
print("\n📊 BASIC STATISTICS FOR NUMERIC COLUMNS:")
stats_df = train_full.select(numeric_cols).describe()
print(stats_df)

# Target variable distribution
if 'y_target' in train_full.columns:
    print("\n🎯 TARGET VARIABLE DISTRIBUTION:")
    target_stats = train_full.select([
        pl.col('y_target').mean().alias('mean'),
        pl.col('y_target').std().alias('std'),
        pl.col('y_target').min().alias('min'),
        pl.col('y_target').max().alias('max'),
        pl.col('y_target').median().alias('median'),
        pl.col('y_target').quantile(0.25).alias('q25'),
        pl.col('y_target').quantile(0.75).alias('q75')
    ])
    print(target_stats)

# ============================================
# MEMORY USAGE ANALYSIS
# ============================================
print("="*60)
print("MEMORY USAGE ANALYSIS")
print("="*60)

# Memory usage for train dataset
train_memory = train_full.estimated_size('mb')
test_memory = test_full.estimated_size('mb')

print(f"\n💾 MEMORY USAGE:")
print(f"Train dataset: {train_memory:.2f} MB")
print(f"Test dataset: {test_memory:.2f} MB")
print(f"Total: {train_memory + test_memory:.2f} MB")

# Memory usage by data type
print("\n📊 MEMORY USAGE BY DATA TYPE:")
dtype_groups = {}
for col in train_full.columns:
    dtype = str(train_full[col].dtype)
    if dtype not in dtype_groups:
        dtype_groups[dtype] = []
    dtype_groups[dtype].append(col)

for dtype, cols in dtype_groups.items():
    subset_memory = train_full.select(cols).estimated_size('mb')
    print(f"{dtype}: {len(cols)} columns, {subset_memory:.2f} MB")

# ============================================
# PERFORMANCE COMPARISON
# ============================================

"""
Key Polars Optimizations Applied:

1. Lazy Loading: Use `pl.scan_parquet()` for large files when possible
2. Column Selection: Select only needed columns to reduce memory
3. Efficient Aggregations: Use Polars' optimized aggregation methods
4. Memory Mapping: Polars automatically uses memory mapping for parquet files
5. Parallel Processing: Polars automatically parallelizes operations

Performance Benefits:
- Faster Loading: Polars reads parquet files more efficiently
- Lower Memory Usage: Better memory management and lazy evaluation
- Faster Aggregations: Optimized group-by and statistical operations
- Better Scaling: Handles large datasets more effectively

Usage Tips:
```python
# For very large datasets, use lazy loading:
# train_lazy = pl.scan_parquet(full_train_path)
# train_filtered = train_lazy.filter(pl.col('y_target').is_not_null()).collect()

# Select only needed columns:
# train_subset = train_full.select(['code', 'y_target', 'feature_a'])

# Use expressions for complex operations:
# train_with_features = train_full.with_columns([
#     (pl.col('feature_a') + pl.col('feature_b')).alias('sum_ab')
# ])
```
"""

print("\n✅ Analysis complete! This Python script provides the same functionality as the Jupyter notebook")
print("   but without the notebook interface overhead and segmentation issues.")
