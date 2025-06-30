import sys
import os
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import FeatureExtractor, AggregateFeatures

def test_feature_extractor():
    """Test time feature extraction"""
    data = pd.DataFrame({
        "TransactionStartTime": ["2023-01-01 12:30:00", "2023-01-02 15:45:00"],
        "CustomerId": [1, 2],
        "Amount": [100, 200],
        "Value": [100, 200],
        "CurrencyCode": ["USD", "EUR"],
        "CountryCode": [1, 2],
        "ProductCategory": ["A", "B"],
        "ChannelId": ["Web", "Mobile"],
        "TransactionId": [1, 2],
        "BatchId": [1, 2],
        "SubscriptionId": [1, 2],
        "ProviderId": [1, 2],
        "ProductId": ["P1", "P2"],
        "PricingStrategy": ["S1", "S2"],
        "FraudResult": [0, 0]
    })
    extractor = FeatureExtractor()
    transformed = extractor.transform(data)
    
    assert "TransactionHour" in transformed.columns
    assert transformed["TransactionHour"].iloc[0] == 12
    assert "TransactionDay" in transformed.columns
    assert transformed.shape[0] == 2

def test_aggregate_features():
    """Test customer aggregation"""
    data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "Amount": [100, 200, 50],
        "Value": [100, 200, 50],
        "TransactionId": [1, 2, 3],
        "BatchId": [1, 1, 2],
        "SubscriptionId": [1, 1, 2],
        "ProviderId": [1, 1, 2],
        "ProductId": ["P1", "P2", "P3"],
        "PricingStrategy": ["S1", "S1", "S2"],
        "FraudResult": [0, 0, 0]
    })
    aggregator = AggregateFeatures()
    transformed = aggregator.transform(data)
    
    assert "TotalAmount" in transformed.columns
    assert "AvgAmount" in transformed.columns
    assert transformed[transformed["CustomerId"] == 1]["TotalAmount"].iloc[0] == 300
    assert transformed.shape[0] == 3