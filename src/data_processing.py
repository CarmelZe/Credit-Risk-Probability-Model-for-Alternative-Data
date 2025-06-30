import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    LabelEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from target_engineering import create_target_variable  # Changed import

class FeatureExtractor(BaseEstimator, TransformerMixin):
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
        ('label_encode', FunctionTransformer(
            lambda x: x.apply(LabelEncoder().fit_transform))
        )
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('time', time_pipeline, time_features)
    ])

def preprocess_data():
    # Load with datetime parsing
    df = pd.read_csv(
        "data/raw/data.csv",  # Updated path
        parse_dates=['TransactionStartTime']
    )
    
    # Create target
    df = create_target_variable(df)
    
    # Feature engineering
    pipeline = Pipeline([
        ('extract_time', FeatureExtractor()),
        ('aggregate', AggregateFeatures()),
        ('process', get_feature_pipeline())
    ])
    
    return pipeline.fit_transform(df)

if __name__ == "__main__":
    processed_data = preprocess_data()
    pd.DataFrame(processed_data).to_csv(
        "data/processed/final_processed_data.csv", 
        index=False
    )
    print("✅ Processing complete!")