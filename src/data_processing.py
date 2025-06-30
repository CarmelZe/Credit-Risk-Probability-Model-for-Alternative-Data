import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Change to relative import
from .target_engineering import create_target_variable

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], errors='coerce')
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour.fillna(0)
        X['TransactionDay'] = X['TransactionStartTime'].dt.day.fillna(1)
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month.fillna(1)
        X['TransactionYear'] = X['TransactionStartTime'].dt.year.fillna(2023)
        return X

class AggregateFeatures(BaseEstimator, TransformerMixin):
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
        return X.merge(agg_features, on='CustomerId', how='left')

def get_feature_pipeline():
    num_features = ['Amount', 'Value', 'TotalAmount', 'AvgAmount', 
                   'TransactionCount', 'AmountStd']
    cat_features = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId']
    time_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    time_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('time', time_pipeline, time_features)
    ])
    
    return Pipeline([
        ('extract_time', FeatureExtractor()),
        ('aggregate', AggregateFeatures()),
        ('preprocess', preprocessor)
    ])

def preprocess_data():
    # Load data with error handling
    try:
        df = pd.read_csv("data/raw/data.csv")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load raw data: {str(e)}")
    
    # Create target variable
    df = create_target_variable(df)
    
    # Feature engineering
    pipeline = get_feature_pipeline()
    processed_data = pipeline.fit_transform(df)
    
    # Get feature names
    feature_names = []
    for name, trans, cols in pipeline.named_steps['preprocess'].transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            cats = pipeline.named_steps['preprocess'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cols)
            feature_names.extend(cats)
        elif name == 'time':
            feature_names.extend(cols)
    
    # Create final DataFrame
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    processed_df['is_high_risk'] = df['is_high_risk'].values
    
    # Final validation
    if processed_df['is_high_risk'].nunique() < 2:
        raise ValueError("Processed data contains only one class in target")
    
    print("✅ Processing complete! Final shape:", processed_df.shape)
    print("Target distribution:\n", processed_df['is_high_risk'].value_counts())
    
    return processed_df

if __name__ == "__main__":
    try:
        processed_data = preprocess_data()
        processed_data.to_csv("data/processed/final_processed_data.csv", index=False)
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")