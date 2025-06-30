import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(data, snapshot_date=None):
    """Calculate RFM metrics with robust datetime handling"""
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], errors='coerce')
    if snapshot_date is None:
        snapshot_date = data['TransactionStartTime'].max()
    
    valid_data = data[data['TransactionStartTime'].notna()]
    
    rfm = valid_data.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    # Higher recency is worse (older), so invert
    rfm['Recency'] = -rfm['Recency']
    return rfm

def cluster_customers(rfm, n_clusters=3, random_state=42):
    """Improved clustering with balanced checks"""
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Run clustering with multiple attempts if needed
    for attempt in range(3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state+attempt)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Check cluster balance
        cluster_counts = rfm['Cluster'].value_counts()
        if cluster_counts.min() > len(rfm)*0.05:  # At least 5% in each cluster
            break
    
    return rfm

def assign_risk_labels(rfm):
    """More robust risk labeling"""
    cluster_means = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean()
    
    # Ensure we have exactly one high-risk cluster
    high_risk_cluster = cluster_means.sum(axis=1).idxmin()
    
    # Force at least 5% high-risk customers
    min_high_risk = int(len(rfm) * 0.05)
    current_high_risk = (rfm['Cluster'] == high_risk_cluster).sum()
    
    if current_high_risk < min_high_risk:
        # If too few, take bottom 5% by Monetary value
        high_risk_idx = rfm.sort_values('Monetary').head(min_high_risk).index
        rfm['Cluster'] = 0  # Default to low-risk
        rfm.loc[high_risk_idx, 'Cluster'] = 1  # Mark as high-risk
        high_risk_cluster = 1
    
    return (rfm['Cluster'] == high_risk_cluster).astype(int)

def create_target_variable(data):
    """Full pipeline with validation"""
    # Calculate RFM
    rfm = calculate_rfm(data)
    
    if len(rfm) < 10:  # Insufficient data check
        raise ValueError("Insufficient customers for clustering")
    
    # Cluster and assign labels
    rfm = cluster_customers(rfm)
    risk_labels = assign_risk_labels(rfm)
    
    # Merge back with original data
    data = data.merge(risk_labels.rename('is_high_risk'), 
                     left_on='CustomerId', 
                     right_index=True,
                     how='left')
    
    # Fill any missing values (new customers) as low-risk
    data['is_high_risk'] = data['is_high_risk'].fillna(0)
    
    # Validation
    target_dist = data['is_high_risk'].value_counts()
    print("Final target distribution:\n", target_dist)
    
    if len(target_dist) < 2:
        raise ValueError("Failed to create meaningful risk classes")
    
    return data

if __name__ == "__main__":
    df = pd.read_csv("data/raw/data.csv")
    df_with_target = create_target_variable(df)
    df_with_target.to_csv("data/processed/data_with_target.csv", index=False)