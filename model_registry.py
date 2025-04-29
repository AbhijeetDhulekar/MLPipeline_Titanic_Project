import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.train import train
from src.preprocessing import preprocess_data  
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REGISTRY_PATH = "models/registry"
MODEL_VERSIONS_PATH = "models"
DATA_STORAGE_PATH = "data/streaming_data"
os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)
os.makedirs(MODEL_VERSIONS_PATH, exist_ok=True)
os.makedirs(DATA_STORAGE_PATH, exist_ok=True)

def get_latest_data(days=7):
    """Combine recent data batches for retraining with validation"""
    try:
        data_files = sorted(Path(DATA_STORAGE_PATH).glob("*.csv"), reverse=True)
        
        if not data_files:
            raise ValueError("No training data available")
        
        # Get files from last N days
        selected_files = data_files[:min(days, len(data_files))]
        logger.info(f"Loading data files: {[f.name for f in selected_files]}")
        
        dfs = []
        for f in selected_files:
            try:
                df = pd.read_csv(f)
                # Validate required columns
                required_cols = {'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Deck', 'Survived'}
                if not required_cols.issubset(df.columns):
                    logger.warning(f"Missing columns in {f.name}, skipping")
                    continue
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {f.name}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No valid data files found")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} records from {len(dfs)} files")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error in get_latest_data: {str(e)}")
        raise

def evaluate_model(model, X, y):
    """Enhanced model evaluation with more metrics"""
    try:
        preds = model.predict(X)
        pred_probs = model.predict_proba(X)[:, 1]
        
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "n_samples": len(y),
            "positive_class_ratio": sum(y) / len(y),
            "features_used": list(X.columns)
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def register_new_version():
    """Enhanced version registration with better error handling"""
    try:
        logger.info("Starting new model version registration")
        
        # 1. Get and preprocess data
        df = get_latest_data()
        logger.info("Data loaded successfully")
        
        # Apply preprocessing
        df = preprocess_data(df)
        logger.info("Data preprocessing completed")
        
        # 2. Prepare features/target
        X = df.drop(['Survived'], axis=1, errors='ignore')
        y = df['Survived']
        
        # Validate data after preprocessing
        if X.empty or y.empty:
            raise ValueError("Empty features or target after preprocessing")
        
        # 3. Retrain model
        logger.info("Starting model training")
        model = train(X, y)
        logger.info("Model training completed")
        
        # 4. Evaluate
        metrics = evaluate_model(model, X, y)
        logger.info(f"Model metrics: {metrics}")
        
        # 5. Create version info
        version_info = {
            "version_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "features": list(X.columns),
            "model_type": str(type(model)),
            "hyperparameters": model.get_params(),
            "data_stats": {
                "n_samples": len(df),
                "start_date": df['timestamp'].min() if 'timestamp' in df.columns else None,
                "end_date": df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }
        
        # 6. Save artifacts
        version_id = version_info["version_id"]
        model_path = f"{MODEL_VERSIONS_PATH}/titanic_model_{version_id}.joblib"
        info_path = f"{MODEL_REGISTRY_PATH}/{version_id}.json"
        
        joblib.dump(model, model_path)
        with open(info_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # 7. Update latest pointer
        joblib.dump(model, f"{MODEL_VERSIONS_PATH}/titanic_model_latest.joblib")
        
        logger.info(f"Successfully registered new model version {version_id}")
        return version_info
        
    except Exception as e:
        logger.error(f"Failed to register new version: {str(e)}")
        return None

def get_model_versions(limit=5):
    """Get registered model versions with optional limit"""
    try:
        version_files = sorted(Path(MODEL_REGISTRY_PATH).glob("*.json"), reverse=True)
        versions = []
        
        for f in version_files[:limit]:
            try:
                with open(f, 'r') as file:
                    versions.append(json.load(file))
            except Exception as e:
                logger.error(f"Error loading version {f.name}: {str(e)}")
                continue
                
        return versions
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        return []

if __name__ == "__main__":
    # Example: Manually trigger retraining with more feedback
    print("Starting manual retraining...")
    result = register_new_version()
    if result:
        print(f"Successfully created version {result['version_id']}")
        print(f"Metrics: {result['metrics']}")
    else:
        print("Retraining failed")