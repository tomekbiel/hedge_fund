import pandas as pd
from pathlib import Path
import pickle
import os

# Cache file paths
CACHE_DIR = Path(r"C:\python\hedge_fund\cache")
CACHE_DIR.mkdir(exist_ok=True)

TEST_CACHE_FILE = CACHE_DIR / "test_data.pkl"
TRAIN_CACHE_FILE = CACHE_DIR / "train_data.pkl"

# Original data paths
TEST_PATH = Path(r"C:\python\hedge_fund\data\test.parquet")
TRAIN_PATH = Path(r"C:\python\hedge_fund\data\train.parquet")

def load_with_cache(data_path, cache_file, data_name):
    """Load data from cache if available, otherwise load and cache it"""
    if cache_file.exists():
        print(f"Loading {data_name} from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Loading {data_name} from parquet and caching...")
        df = pd.read_parquet(data_path)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"{data_name} cached to {cache_file}")
        return df

# Load test data (with caching)
df_test = load_with_cache(TEST_PATH, TEST_CACHE_FILE, "test data")

# Load train data (with caching)
df_train = load_with_cache(TRAIN_PATH, TRAIN_CACHE_FILE, "train data")

print("=== TEST DATA ===")
print(df_test.head())
print("Columns:", df_test.columns.tolist())
print(df_test.info())
print("Shape:", df_test.shape)
print(df_test.describe())

print("\n=== TRAIN DATA ===")
print(df_train.head())
print("Columns:", df_train.columns.tolist())
print(df_train.info())
print("Shape:", df_train.shape)
print(df_train.describe())
