#!/usr/bin/env python
# coding: utf-8

# # FULL DATASET ANALYSIS
# 
# This notebook contains comprehensive analysis for the full datasets
# 
# **Datasets:**
# - train_full: 5,337,414 rows, 94 columns
# - test_full: 1,447,107 rows, 92 columns
# 
# **Analysis Pipeline:**
# 1. Data identification and structure
# 2. Missing values analysis
# 3. Distribution analysis (all types)
# 4. Outlier detection
# 5. Data type optimization
# 6. Standardization strategies

# In[1]:


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

# Display configuration - show all rows and columns

# Set up paths
base_dir = Path("..")
full_train_path = base_dir / "data" / "train.parquet"
full_test_path = base_dir / "data" / "test.parquet"

print("✅ Imports and configuration complete")


# 

# In[2]:


# ============================================
# LOAD FULL DATASETS
# ============================================
print("="*60)
print("LOADING FULL DATASETS")
print("="*60)

# Load full datasets
train_full = pl.read_parquet(full_train_path)
test_full = pl.read_parquet(full_test_path)

print(f"Train full: {train_full.shape}")
print(f"Test full: {test_full.shape}")

print(f"\n✅ Full datasets loaded")
print(f"   Train: {train_full.shape}")
print(f"   Test: {test_full.shape}")


# ## STEP 1: DATA IDENTIFICATION AND STRUCTURE

# In[3]:


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
print(train_full.dtypes)

# Test dataset structure  
print("\n📋 TEST DATASET STRUCTURE:")
print(f"Shape: {test_full.shape}")
print(f"Columns: {test_full.columns}")
print(f"\nData types:")
print(test_full.dtypes)

# Column differences
train_cols = set(train_full.columns)
test_cols = set(test_full.columns)
only_in_train = train_cols - test_cols
only_in_test = test_cols - train_cols

print(f"\n📊 COLUMN DIFFERENCES:")
print(f"Only in train: {only_in_train}")
print(f"Only in test: {only_in_test}")
print(f"Common columns: {len(train_cols & test_cols)}")


# In[4]:


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
            print(train_full[col].value_counts())

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

