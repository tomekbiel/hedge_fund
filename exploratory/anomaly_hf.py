# ============================================
# SEGMENT 1: All Imports
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For K-Means and LOF
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For baseline model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Plotting settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully")


# ============================================
# SEGMENT 2: Data Loading and Initial Analysis
# ============================================

# Load data (adjust the path to your file)
data_path = Path("C:/python/hedge_fund/data/sample/train_sample.parquet")
df = pd.read_parquet(data_path)

print("=" * 60)
print("📊 BASIC INFORMATION ABOUT THE DATA")
print("=" * 60)
print(f"Data shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nData preview:")
print(df.head())

# Check for missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'missing': missing,
    'percentage': missing_pct
}).sort_values('percentage', ascending=False)

print("\n🔍 TOP 10 COLUMNS WITH MOST MISSING VALUES:")
print(missing_df.head(10))

# Basic statistics for numerical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n📈 Number of numerical columns: {len(numeric_cols)}")
print(f"First 5: {numeric_cols[:5]}")

# Check for constant or near-constant columns
print("\n📉 CHECKING FOR CONSTANT OR NEAR-CONSTANT COLUMNS:")
constant_cols = []
for col in numeric_cols:
    unique_ratio = df[col].nunique() / len(df)
    if unique_ratio < 0.01:  # Less than 1% unique values
        constant_cols.append(col)
        print(f"  {col}: {unique_ratio:.4f} unique ratio")

if constant_cols:
    print(f"\n⚠️ Found {len(constant_cols)} columns with very low cardinality")
else:
    print("✅ No constant or near-constant columns found")

# Check data types of categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n📊 Categorical columns: {categorical_cols}")
for col in categorical_cols:
    print(f"  {col}: {df[col].nunique()} unique values")
    if df[col].nunique() < 20:  # Show value counts for small cardinality columns
        print(f"    Value counts: {df[col].value_counts().to_dict()}")

# ============================================
# SEGMENT 2B: Target Variable Analysis
# ============================================

print("=" * 60)
print("🎯 TARGET VARIABLE (y_target) ANALYSIS")
print("=" * 60)

# Basic statistics of target
target_stats = df['y_target'].describe()
print("\nBasic statistics:")
print(target_stats)

# Check for skewness and kurtosis
skewness = df['y_target'].skew()
kurtosis = df['y_target'].kurtosis()
print(f"\nSkewness: {skewness:.4f} (0 = normal, >0 = right-skewed, <0 = left-skewed)")
print(f"Kurtosis: {kurtosis:.4f} (3 = normal, >3 = heavy tails, <3 = light tails)")

# Visualization of target distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Histogram
axes[0, 0].hist(df['y_target'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Target Distribution - Histogram')
axes[0, 0].set_xlabel('y_target')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Boxplot
axes[0, 1].boxplot(df['y_target'])
axes[0, 1].set_title('Target Distribution - Boxplot')
axes[0, 1].set_ylabel('y_target')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Density plot
df['y_target'].plot(kind='density', ax=axes[0, 2], color='steelblue', linewidth=2)
axes[0, 2].set_title('Target Distribution - Density Plot')
axes[0, 2].set_xlabel('y_target')
axes[0, 2].grid(True, alpha=0.3)

# Q-Q plot - NOW USING stats WHICH IS IMPORTED
stats.probplot(df['y_target'].dropna(), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot vs Normal Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Target vs ts_index (time series view)
axes[1, 1].scatter(df['ts_index'], df['y_target'], alpha=0.3, s=1, color='steelblue')
axes[1, 1].set_title('Target Over Time')
axes[1, 1].set_xlabel('ts_index')
axes[1, 1].set_ylabel('y_target')
axes[1, 1].grid(True, alpha=0.3)

# Target by horizon
horizon_means = df.groupby('horizon')['y_target'].mean()
horizon_std = df.groupby('horizon')['y_target'].std()
horizons = sorted(df['horizon'].unique())

axes[1, 2].bar(horizons, horizon_means, yerr=horizon_std, capsize=5,
               color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Target Mean ± Std by Horizon')
axes[1, 2].set_xlabel('Horizon')
axes[1, 2].set_ylabel('Mean y_target')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Detect potential outliers in target using IQR
Q1 = df['y_target'].quantile(0.25)
Q3 = df['y_target'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

target_outliers = df[(df['y_target'] < lower_bound) | (df['y_target'] > upper_bound)]
print(f"\n🔍 Target outliers (IQR method):")
print(f"  Lower bound: {lower_bound:.4f}")
print(f"  Upper bound: {upper_bound:.4f}")
print(f"  Number of outliers: {len(target_outliers)} ({len(target_outliers)/len(df)*100:.2f}%)")

# ============================================
# SEGMENT 2C: Time Series Structure Analysis
# ============================================

print("=" * 60)
print("📅 TIME SERIES STRUCTURE ANALYSIS")
print("=" * 60)

# ts_index analysis
print(f"\nts_index range: {df['ts_index'].min()} to {df['ts_index'].max()}")
print(f"Unique ts_index values: {df['ts_index'].nunique()}")
print(f"Expected vs actual: Range should be {df['ts_index'].max() - df['ts_index'].min() + 1}, "
      f"actual unique: {df['ts_index'].nunique()}")

# Check if we have all combinations
print("\n📊 Data completeness by groups:")

# By code
code_completeness = df.groupby('code').size().describe()
print(f"\nRows per code:\n{code_completeness}")

# By sub_category
cat_completeness = df.groupby('sub_category').size()
print(f"\nRows per sub_category:\n{cat_completeness}")

# By horizon
horizon_completeness = df.groupby('horizon').size()
print(f"\nRows per horizon:\n{horizon_completeness}")

# Check if we have all ts_index for each combination
print("\n🔍 Checking for missing time points...")

# Create a sample of one code to check continuity
sample_code = df['code'].iloc[0]
sample_data = df[df['code'] == sample_code].sort_values('ts_index')

print(f"\nSample code '{sample_code}':")
print(f"  Number of records: {len(sample_data)}")
print(f"  ts_index range: {sample_data['ts_index'].min()} to {sample_data['ts_index'].max()}")
print(f"  Unique ts_index: {sample_data['ts_index'].nunique()}")
print(f"  Unique horizons: {sample_data['horizon'].unique()}")

# Visualization of time series structure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribution of ts_index
axes[0, 0].hist(df['ts_index'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of ts_index')
axes[0, 0].set_xlabel('ts_index')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Number of records per ts_index
records_per_time = df.groupby('ts_index').size()
axes[0, 1].plot(records_per_time.index, records_per_time.values, 'o-', markersize=3)
axes[0, 1].set_title('Number of Records per ts_index')
axes[0, 1].set_xlabel('ts_index')
axes[0, 1].set_ylabel('Number of records')
axes[0, 1].grid(True, alpha=0.3)

# Target over time for a few sample codes
sample_codes = df['code'].unique()[:5]
for code in sample_codes:
    code_data = df[df['code'] == code].sort_values('ts_index')
    axes[1, 0].plot(code_data['ts_index'], code_data['y_target'], 'o-',
                   markersize=2, linewidth=1, alpha=0.7, label=code[:10])
axes[1, 0].set_title('Target Over Time - Sample Codes')
axes[1, 0].set_xlabel('ts_index')
axes[1, 0].set_ylabel('y_target')
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 0].grid(True, alpha=0.3)

# Heatmap of availability (code vs ts_index) - sampled for visualization
# Take a sample of codes and ts_indices for visualization
sample_codes_small = df['code'].unique()[:20]
sample_df = df[df['code'].isin(sample_codes_small)]
pivot_avail = sample_df.pivot_table(
    index='code',
    columns='ts_index',
    values='y_target',
    aggfunc='count',
    fill_value=0
)
# Sample ts_indices for better visualization
if pivot_avail.shape[1] > 50:
    ts_sample = np.linspace(pivot_avail.columns.min(),
                           pivot_avail.columns.max(), 50).astype(int)
    pivot_avail = pivot_avail[[col for col in pivot_avail.columns if col in ts_sample]]

sns.heatmap(pivot_avail > 0, ax=axes[1, 1], cbar=False, cmap='Blues')
axes[1, 1].set_title('Data Availability (sampled)')
axes[1, 1].set_xlabel('ts_index')
axes[1, 1].set_ylabel('code')

plt.tight_layout()
plt.show()


# ============================================
# SEGMENT 2D: Weight Column Analysis
# ============================================

print("=" * 60)
print("⚖️ WEIGHT COLUMN ANALYSIS")
print("=" * 60)

# Basic statistics of weights
weight_stats = df['weight'].describe()
print("\nWeight statistics:")
print(weight_stats)

# Check if weights sum to something meaningful
print(f"\nSum of weights: {df['weight'].sum():.4f}")
print(f"Weights range: {df['weight'].min():.4f} to {df['weight'].max():.4f}")

# Check distribution of weights
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram of weights
axes[0].hist(df['weight'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Weight Distribution')
axes[0].set_xlabel('weight')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Weights by horizon
df.boxplot(column='weight', by='horizon', ax=axes[1])
axes[1].set_title('Weights by Horizon')
axes[1].set_xlabel('horizon')
axes[1].set_ylabel('weight')
plt.suptitle('')  # Remove automatic suptitle

# Weights by sub_category
cat_weights = df.groupby('sub_category')['weight'].mean().sort_values()
axes[2].barh(range(len(cat_weights)), cat_weights.values)
axes[2].set_yticks(range(len(cat_weights)))
axes[2].set_yticklabels(cat_weights.index)
axes[2].set_title('Mean Weight by Sub-category')
axes[2].set_xlabel('Mean weight')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Correlation between weight and target
corr_weight_target = df['weight'].corr(df['y_target'])
print(f"\nCorrelation between weight and target: {corr_weight_target:.4f}")

# Check if weights are related to any patterns
print("\nWeight statistics by quantiles of target:")
df['target_quantile'] = pd.qcut(df['y_target'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
weight_by_target_quantile = df.groupby('target_quantile')['weight'].agg(['mean', 'std', 'count'])
print(weight_by_target_quantile)

# ============================================
# SEGMENT 2E: Feature Correlation Analysis
# ============================================

print("=" * 60)
print("🔗 FEATURE CORRELATION ANALYSIS")
print("=" * 60)

# Select only feature columns (excluding metadata and target)
feature_cols = [col for col in numeric_cols if col not in ['ts_index', 'horizon', 'y_target', 'weight']]
print(f"Analyzing {len(feature_cols)} feature columns")

# Calculate correlation matrix (with target)
corr_with_target = pd.DataFrame({
    'feature': feature_cols,
    'correlation_with_target': [df[col].corr(df['y_target']) for col in feature_cols]
}).sort_values('correlation_with_target', ascending=False)

print("\n📈 TOP 10 FEATURES CORRELATED WITH TARGET (positive):")
print(corr_with_target.head(10))

print("\n📉 TOP 10 FEATURES CORRELATED WITH TARGET (negative):")
print(corr_with_target.tail(10))

# Correlation matrix of features (sample due to size)
if len(feature_cols) > 30:
    # Take top correlated with target for visualization
    top_features = corr_with_target.head(15)['feature'].tolist() + \
                   corr_with_target.tail(15)['feature'].tolist()
    top_features = list(set(top_features))  # Remove duplicates
else:
    top_features = feature_cols

print(f"\nVisualizing correlation matrix for {len(top_features)} features")

# Calculate correlation matrix
corr_matrix = df[top_features].corr()

# Visualize
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Matrix (top features)')
plt.tight_layout()
plt.show()

# Check for highly correlated feature pairs
print("\n🔍 HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.8):")
high_corr_pairs = []
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.8:
            high_corr_pairs.append({
                'feature1': top_features[i],
                'feature2': top_features[j],
                'correlation': corr
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    print(high_corr_df)
else:
    print("No highly correlated pairs found")

# ============================================
# SEGMENT 2F: Missing Data Pattern Analysis
# ============================================

print("=" * 60)
print("❓ MISSING DATA PATTERN ANALYSIS")
print("=" * 60)

# Calculate missing percentage for each column
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'column': missing_pct.index,
    'missing_pct': missing_pct.values
}).sort_values('missing_pct', ascending=False)

print("\nColumns with missing values:")
print(missing_df[missing_df['missing_pct'] > 0])

# Visualize missing patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Bar plot of missing percentages
top_missing = missing_df[missing_df['missing_pct'] > 0].head(20)
if len(top_missing) > 0:
    axes[0, 0].barh(range(len(top_missing)), top_missing['missing_pct'].values)
    axes[0, 0].set_yticks(range(len(top_missing)))
    axes[0, 0].set_yticklabels(top_missing['column'].values)
    axes[0, 0].set_xlabel('Missing Percentage (%)')
    axes[0, 0].set_title('Top 20 Columns with Missing Values')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
else:
    axes[0, 0].text(0.5, 0.5, 'No missing values', ha='center', va='center')

# Missing pattern over time (ts_index)
if len(df[df.isnull().any(axis=1)]) > 0:
    missing_over_time = df.isnull().any(axis=1).groupby(df['ts_index']).mean() * 100
    axes[0, 1].plot(missing_over_time.index, missing_over_time.values)
    axes[0, 1].set_xlabel('ts_index')
    axes[0, 1].set_ylabel('Percentage of rows with missing values')
    axes[0, 1].set_title('Missing Data Over Time')
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'No missing values', ha='center', va='center')

# Missing by horizon
if len(df[df.isnull().any(axis=1)]) > 0:
    missing_by_horizon = df.isnull().any(axis=1).groupby(df['horizon']).mean() * 100
    axes[1, 0].bar(missing_by_horizon.index, missing_by_horizon.values)
    axes[1, 0].set_xlabel('horizon')
    axes[1, 0].set_ylabel('Percentage of rows with missing values')
    axes[1, 0].set_title('Missing Data by Horizon')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
else:
    axes[1, 0].text(0.5, 0.5, 'No missing values', ha='center', va='center')

# Missing by sub_category
if len(df[df.isnull().any(axis=1)]) > 0:
    missing_by_cat = df.isnull().any(axis=1).groupby(df['sub_category']).mean() * 100
    axes[1, 1].barh(range(len(missing_by_cat)), missing_by_cat.values)
    axes[1, 1].set_yticks(range(len(missing_by_cat)))
    axes[1, 1].set_yticklabels(missing_by_cat.index)
    axes[1, 1].set_xlabel('Percentage of rows with missing values')
    axes[1, 1].set_title('Missing Data by Sub-category')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
else:
    axes[1, 1].text(0.5, 0.5, 'No missing values', ha='center', va='center')

plt.tight_layout()
plt.show()

# Summary statistics
total_missing_cells = df.isnull().sum().sum()
total_cells = df.size
print(f"\n📊 MISSING DATA SUMMARY:")
print(f"Total missing cells: {total_missing_cells}")
print(f"Total cells: {total_cells}")
print(f"Overall missing percentage: {total_missing_cells/total_cells*100:.4f}%")
print(f"Rows with at least one missing: {df.isnull().any(axis=1).sum()} ({df.isnull().any(axis=1).mean()*100:.2f}%)")


