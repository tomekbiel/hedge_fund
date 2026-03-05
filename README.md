# Hedge Fund Data Analysis Project

## Project Structure

```
hedge_fund/
├── data/
│   ├── train.parquet          # Full training data (776MB)
│   ├── test.parquet           # Full test data (146MB)
│   └── sample/                # Sample data for GitHub
│       ├── train_sample.parquet    # 1% sample (36MB)
│       ├── test_sample.parquet     # 1% sample (10MB)
│       └── data_info.txt          # Dataset information
├── exploratory/
│   ├── data_check.py         # Data exploration with caching
│   ├── efficient_data_loader.py  # Optimized data loading for ML
│   └── create_sample_data.py     # Script to create sample datasets
└── cache/                    # Cached data files (auto-generated)
```

## Quick Start

### 1. Using Sample Data (Recommended for GitHub users)
```python
from exploratory.efficient_data_loader import EfficientDataLoader

loader = EfficientDataLoader("data/sample")
train_df, test_df = loader.load_full_data()
```

### 2. Using Full Data (If you have the complete dataset)
```python
from exploratory.efficient_data_loader import EfficientDataLoader

loader = EfficientDataLoader("data")
train_df, test_df = loader.load_full_data()
```

### 3. Data Exploration with Caching
```python
# Run data exploration script
python exploratory/data_check.py
```

## Data Information

- **Original Dataset**: 
  - Train: 5,337,414 rows × 94 columns
  - Test: 1,447,107 rows × 92 columns
- **Sample Dataset**:
  - Train: 53,374 rows × 94 columns (1% of original)
  - Test: 14,471 rows × 92 columns (1% of original)

## Features

### Efficient Data Loading
- **Caching**: Automatically caches loaded data to speed up subsequent runs
- **Sampling**: Load small portions of data for quick testing
- **Memory Optimization**: Converts data to optimal types for ML models

### Data Processing
- **XGBoost Preparation**: Automatic conversion of categorical variables
- **Type Optimization**: Uses float32 instead of float64 for memory efficiency
- **Flexible Loading**: Support for both full and sample datasets

## Getting Full Dataset

The complete dataset is too large for GitHub. To get the full data:

1. **Download from original source** (if available)
2. **Contact project maintainer** for data access
3. **Use sample data** for development and testing

## Development Workflow

1. **Start with sample data** for quick prototyping
2. **Use caching** to avoid repeated data loading
3. **Scale to full data** only for final model training
4. **Optimize memory usage** with efficient data types

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## Performance Tips

- Use `loader.load_sample(sample_size=0.05)` for quick testing
- Enable caching for repeated experiments
- Use float32 data types for ML models
- Consider Dask for datasets larger than available RAM
