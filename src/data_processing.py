# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime

# --- Step 2: Feature Extraction Class ---
class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features from TransactionStartTime."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

# --- Step 2: Aggregate Features Class ---
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Creates customer-level aggregate features."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        agg_features = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
            'Value': ['sum', 'mean']
        })
        agg_features.columns = [
            'TotalAmount', 'AvgAmount', 'TransactionCount', 'AmountStd',
            'TotalValue', 'AvgValue'
        ]
        X = X.merge(agg_features, on='CustomerId', how='left')
        return X

# --- Step 3: Main Pipeline Function ---
def get_feature_pipeline():
    """
    Returns a scikit-learn Pipeline for end-to-end feature engineering.
    Includes:
    - Time feature extraction
    - Customer aggregates
    - Missing value imputation
    - Scaling (numerical) and encoding (categorical)
    """
    # Numerical features pipeline
    num_features = ['Amount', 'Value', 'TotalAmount', 'AvgAmount', 
                    'TransactionCount', 'AmountStd']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Step 4: Handle missing values
        ('scaler', StandardScaler())                   # Step 5: Standardize
    ])
    
    # Categorical features pipeline
    cat_features = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId']
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Step 4: Impute
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Step 3: Encode
    ])
    
    # Time features pipeline (label-encoded)
    time_features = ['TransactionHour', 'TransactionDay', 
                    'TransactionMonth', 'TransactionYear']
    time_pipeline = Pipeline([
        ('label_encode', FunctionTransformer(
            lambda x: x.apply(LabelEncoder().fit_transform))
        )
    ])
    
    # Combine all pipelines
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('time', time_pipeline, time_features)
    ])
    
    # Final pipeline with feature extraction + aggregation
    pipeline = Pipeline([
        ('extract_time', FeatureExtractor()),      # Step 2: Extract time features
        ('aggregate', AggregateFeatures()),        # Step 2: Create aggregates
        ('process', full_pipeline)                # Step 3-5: Impute/Scale/Encode
    ])
    
    return pipeline

# --- Step 6: Example Usage ---
if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("../data/raw/data.csv")
    
    # Run pipeline
    pipeline = get_feature_pipeline()
    processed_data = pipeline.fit_transform(df)
    
    # Save processed data
    pd.DataFrame(processed_data).to_csv("../data/processed/processed_data.csv", index=False)
    print("✅ Feature engineering complete! Saved to data/processed/")