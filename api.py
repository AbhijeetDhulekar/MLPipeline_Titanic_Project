from fastapi import FastAPI, HTTPException
import joblib
from src.schemas import Passenger
from src.preprocessing import preprocess_data
import pandas as pd
import numpy as np
import uvicorn
from pathlib import Path
import json

app = FastAPI()

MODEL_VERSIONS_PATH = "models"
MODEL_REGISTRY_PATH = "models/registry"  # Define the registry path

def load_latest_model():
    model_files = sorted(Path(MODEL_VERSIONS_PATH).glob("titanic_model_*.joblib"), reverse=True)
    if not model_files:
        raise ValueError("No models found")

    latest_model = model_files[0]
    print(f"Loading model: {latest_model.name}")
    return joblib.load(latest_model)

# Update your model loading
try:
    model = load_latest_model()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/predict")
async def predict(passenger: Passenger):
    try:
        # Convert to DataFrame
        passenger_data = passenger.dict()
        df = pd.DataFrame([passenger_data])

        # Ensure all expected columns exist and handle potential 'Deck' mapping
        if 'Deck' in df.columns and 'Cabin' not in df.columns:
            df['Cabin'] = df['Deck']  # Map Deck to Cabin if needed

        # Preprocess
        processed = preprocess_data(df)

        # Predict
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        return {
            "survived": bool(prediction),
            "probability": float(probability),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health():
    return {
        "status": "OK",
        "model_loaded": model is not None
    }

@app.get("/model-versions")
async def list_model_versions():
    version_files = sorted(Path(MODEL_REGISTRY_PATH).glob("*.json"), reverse=True)
    versions = []

    for f in version_files:
        try:
            with open(f, 'r') as file:
                versions.append(json.load(file))
        except Exception as e:
            print(f"Error reading version file {f.name}: {e}")

    return versions

if __name__ == "__main__":
    uvicorn.run("api:app", reload=True)