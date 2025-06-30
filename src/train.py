import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from data_processing import preprocess_data

def load_data():
    """Load and validate processed data"""
    try:
        data = preprocess_data()
        
        # Validate target
        if 'is_high_risk' not in data.columns:
            raise ValueError("Target column missing")
        if data['is_high_risk'].isna().any():
            print("⚠️ NaN values in target, filling with 0")
            data['is_high_risk'] = data['is_high_risk'].fillna(0)
        
        return data
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        print("Debug steps:")
        print("1. Check data/raw/data.csv exists")
        print("2. Run data_processing.py separately")
        raise

def train_models(X_train, y_train):
    """Train models with class weight balancing"""
    models = {
        "Random Forest": RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
    }

    params = {
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        "Logistic Regression": {"C": [0.1, 1, 10]}
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Hyperparameter tuning
            clf = GridSearchCV(model, params[name], cv=3, scoring='roc_auc')
            clf.fit(X_train, y_train)

            # Log metrics
            mlflow.log_params(clf.best_params_)
            mlflow.sklearn.log_model(clf.best_estimator_, name)

            # Track best model
            if clf.best_score_ > best_score:
                best_model = clf.best_estimator_
                best_score = clf.best_score_

    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Log to MLflow
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

    return metrics

if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Credit_Risk_Model")

    try:
        # Load and split data
        data = load_data()
        X = data.drop("is_high_risk", axis=1)
        y = data["is_high_risk"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train and evaluate
        best_model = train_models(X_train, y_train)
        metrics = evaluate_model(best_model, X_test, y_test)

        # Register best model
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="CreditRiskModel"
        )

        print(f"✅ Training complete! Best model metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Debugging steps:")
        print("1. Verify data_processing.py returns DataFrame with 'is_high_risk'")
        print("2. Check column names in processed data")