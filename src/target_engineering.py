import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(data, snapshot_date=None):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    
    if snapshot_date is None:
        snapshot_date = data['TransactionStartTime'].max()
    
    rfm = data.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    rfm['Recency'] = -rfm['Recency']
    return rfm

def cluster_customers(rfm, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm

def assign_risk_labels(rfm):
    cluster_means = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_means.sum(axis=1).idxmin()
    return (rfm['Cluster'] == high_risk_cluster).astype(int)

def create_target_variable(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    rfm = calculate_rfm(data)
    rfm = cluster_customers(rfm)
    data['is_high_risk'] = assign_risk_labels(rfm)
    return data

if __name__ == "__main__":
    df = pd.read_csv(
        "data/raw/data.csv",
        parse_dates=['TransactionStartTime']
    )
    df_with_target = create_target_variable(df)
    df_with_target.to_csv("data/processed/data_with_target.csv", index=False)
    print("✅ Target variable created!")