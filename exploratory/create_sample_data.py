import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def create_sample_data(input_dir, output_dir, sample_size=0.01, random_state=42):
    """
    Create sample dataset from full data for GitHub upload
    
    Args:
        input_dir: Directory with full parquet files
        output_dir: Directory to save sample files
        sample_size: Fraction of data to sample (default 1%)
        random_state: Random seed for reproducibility
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating {sample_size*100}% sample data...")
    
    # Process train data
    print("Processing train data...")
    train_file = input_path / "train.parquet"
    if train_file.exists():
        df_train = pd.read_parquet(train_file)
        print(f"Original train data shape: {df_train.shape}")
        
        # Sample the data
        if sample_size < 1.0:
            df_train_sample = df_train.sample(frac=sample_size, random_state=random_state)
        else:
            df_train_sample = df_train
            
        print(f"Sample train data shape: {df_train_sample.shape}")
        
        # Save sample
        train_output = output_path / "train_sample.parquet"
        df_train_sample.to_parquet(train_output, index=False)
        print(f"Saved: {train_output} ({train_output.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Process test data
    print("\nProcessing test data...")
    test_file = input_path / "test.parquet"
    if test_file.exists():
        df_test = pd.read_parquet(test_file)
        print(f"Original test data shape: {df_test.shape}")
        
        # Sample the data
        if sample_size < 1.0:
            df_test_sample = df_test.sample(frac=sample_size, random_state=random_state)
        else:
            df_test_sample = df_test
            
        print(f"Sample test data shape: {df_test_sample.shape}")
        
        # Save sample
        test_output = output_path / "test_sample.parquet"
        df_test_sample.to_parquet(test_output, index=False)
        print(f"Saved: {test_output} ({test_output.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Create data info file
    info_file = output_path / "data_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Sample Dataset Information\n")
        f.write(f"========================\n")
        f.write(f"Sample size: {sample_size*100}%\n")
        f.write(f"Random seed: {random_state}\n")
        f.write(f"Original data location: {input_path}\n")
        f.write(f"Created: {pd.Timestamp.now()}\n\n")
        
        if 'df_train_sample' in locals():
            f.write(f"Train sample shape: {df_train_sample.shape}\n")
            f.write(f"Train columns: {list(df_train_sample.columns)}\n\n")
        
        if 'df_test_sample' in locals():
            f.write(f"Test sample shape: {df_test_sample.shape}\n")
            f.write(f"Test columns: {list(df_test_sample.columns)}\n")
    
    print(f"\nSample data created in: {output_path}")
    print(f"Data info saved: {info_file}")
    
    return df_train_sample if 'df_train_sample' in locals() else None, \
           df_test_sample if 'df_test_sample' in locals() else None

def create_various_sample_sizes(input_dir, output_dir, sizes=[0.001, 0.005, 0.01, 0.05]):
    """Create multiple sample sizes for different use cases"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Creating {size*100}% sample...")
        
        size_dir = output_path / f"sample_{int(size*1000)}per_mille"
        create_sample_data(input_path, size_dir, sample_size=size)
    
    print(f"\nAll samples created in: {output_path}")

def optimize_for_github(df, max_size_mb=50):
    """
    Optimize dataframe to stay under GitHub size limit
    
    Args:
        df: DataFrame to optimize
        max_size_mb: Maximum file size in MB
    """
    # Convert to smaller data types
    df_optimized = df.copy()
    
    # Numeric columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        if df_optimized[col].min() >= 0:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='unsigned')
        else:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Categorical columns
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # If cardinality is low
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = r"C:\python\hedge_fund\data"
    OUTPUT_DIR = r"C:\python\hedge_fund\data\sample"
    
    # Create 1% sample (good for GitHub)
    train_sample, test_sample = create_sample_data(
        INPUT_DIR, 
        OUTPUT_DIR, 
        sample_size=0.01  # 1% of data
    )
    
    # Optional: Create multiple sample sizes
    # create_various_sample_sizes(INPUT_DIR, OUTPUT_DIR)
    
    print("\nDone! Upload the sample files to GitHub.")
    print("Files to upload:")
    print("- data/sample/train_sample.parquet")
    print("- data/sample/test_sample.parquet")
