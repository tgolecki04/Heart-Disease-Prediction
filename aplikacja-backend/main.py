from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import pandas as pd
import numpy as np
import traceback

# --- Optional Imports for Neural Networks ---
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")

# Feature order MUST match the order used during training (feature_names list)
EXPECTED_FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

app = FastAPI(title="Medical Heart Risk API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


class PredictRequest(BaseModel):
    # Defining fields to match the HTML form
    male: float
    age: float
    currentSmoker: float
    cigsPerDay: float
    BPMeds: float
    prevalentStroke: float
    prevalentHyp: float
    diabetes: float
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float


# Global Storage
ARTIFACTS = {
    "model": None,
    "scaler": None,
    "imputer": None,
    "threshold": 0.5,
    "type": "unknown"
}


@app.on_event("startup")
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return

    try:
        # Load the full dictionary
        data = joblib.load(MODEL_PATH)
        print("File loaded successfully.")

        # 1. Extract Model
        if isinstance(data, dict):
            # Support your specific saving structure
            ARTIFACTS["model"] = data.get("model")
            ARTIFACTS["scaler"] = data.get("scaler")
            ARTIFACTS["imputer"] = data.get("imputer")
            ARTIFACTS["threshold"] = data.get("optimal_threshold", 0.5)

            # Debug prints
            print(f"Keys found: {list(data.keys())}")
            if ARTIFACTS["scaler"]: print("✅ Scaler loaded")
            if ARTIFACTS["imputer"]: print("✅ Imputer loaded")
            print(f"✅ Optimal Threshold loaded: {ARTIFACTS['threshold']}")
        else:
            # Fallback if just the model was saved
            ARTIFACTS["model"] = data

        # 2. Identify Model Type
        m = ARTIFACTS["model"]
        if hasattr(m, "predict"):
            ARTIFACTS["type"] = "sklearn"  # Generic (RF, XGB, etc)

        if keras and isinstance(m, keras.Model):
            ARTIFACTS["type"] = "keras"

        if torch and isinstance(m, nn.Module):
            ARTIFACTS["type"] = "torch"
            m.eval()  # Set to eval mode immediately

        print(f"Model Type Detected: {ARTIFACTS['type']}")

    except Exception as e:
        print(f"CRITICAL LOAD ERROR: {e}")
        traceback.print_exc()


@app.post("/predict")
def predict(req: PredictRequest):
    if ARTIFACTS["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # 1. Prepare Data (DataFrame)
        input_data = {k: float(getattr(req, k)) for k in EXPECTED_FEATURES}
        df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)

        # 2. Apply Preprocessing (Imputer -> Scaler)
        # IMPORTANT: Convert to numpy array often happens here
        x = df

        if ARTIFACTS["imputer"]:
            x = ARTIFACTS["imputer"].transform(x)

        if ARTIFACTS["scaler"]:
            x = ARTIFACTS["scaler"].transform(x)

        # 3. Predict
        model = ARTIFACTS["model"]
        m_type = ARTIFACTS["type"]
        threshold = ARTIFACTS["threshold"]

        prob = 0.0

        # --- PyTorch ---
        if m_type == "torch":
            # Convert to Tensor (float32)
            tensor_in = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_in)
                # Apply Sigmoid if output is logits (usually 1 output neuron)
                val = logits.item()
                prob = 1 / (1 + np.exp(-val))  # Sigmoid

        # --- Keras ---
        elif m_type == "keras":
            # Keras expects numpy array
            raw = model.predict(x, verbose=0)
            if raw.shape[-1] == 1:
                prob = float(raw[0][0])
            else:
                prob = float(raw[0][1])  # Softmax class 1

        # --- Sklearn / XGBoost ---
        else:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(x)[0][1])
            else:
                # Fallback for models without proba
                prob = float(model.predict(x)[0])

        # 4. Determine Class based on Optimal Threshold
        is_risk = 1 if prob >= threshold else 0

        return {
            "class": is_risk,
            "probability": prob,
            "threshold_used": threshold
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": ARTIFACTS["model"] is not None,
        "type": ARTIFACTS["type"],
        "has_scaler": ARTIFACTS["scaler"] is not None
    }