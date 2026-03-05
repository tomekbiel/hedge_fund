import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

class EfficientDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.test_path = self.data_dir / "test.parquet"
        self.train_path = self.data_dir / "train.parquet"
        
    def load_sample(self, sample_size=0.1, random_state=42):
        """Load sample of data for quick testing"""
        print(f"Loading {sample_size*100}% sample of data...")
        
        # Load train sample
        train_df = pd.read_parquet(self.train_path)
        if sample_size < 1.0:
            train_df = train_df.sample(frac=sample_size, random_state=random_state)
        
        # Load test sample  
        test_df = pd.read_parquet(self.test_path)
        if sample_size < 1.0:
            test_df = test_df.sample(frac=sample_size, random_state=random_state)
            
        return train_df, test_df
    
    def load_full_data(self):
        """Load full dataset - use for final training"""
        print("Loading full dataset...")
        train_df = pd.read_parquet(self.train_path)
        test_df = pd.read_parquet(self.test_path)
        return train_df, test_df
    
    def prepare_for_xgboost(self, df, target_column=None):
        """Optimize data for XGBoost"""
        # Convert categorical to numeric (XGBoost requirement)
        df_processed = df.copy()
        
        # Handle categorical columns
        cat_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in cat_columns:
            df_processed[col] = pd.Categorical(df_processed[col]).codes
            
        # Convert to float32 (XGBoost prefers float32)
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].astype(np.float32)
        
        if target_column and target_column in df_processed.columns:
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
            return X, y
        
        return df_processed

# Usage example:
if __name__ == "__main__":
    loader = EfficientDataLoader(r"C:\python\hedge_fund\data")
    
    # For quick development/testing
    train_sample, test_sample = loader.load_sample(sample_size=0.05)
    
    # For final model training
    # train_full, test_full = loader.load_full_data()
    
    # Prepare for XGBoost
    # X_train, y_train = loader.prepare_for_xgboost(train_sample, target_column='target')
