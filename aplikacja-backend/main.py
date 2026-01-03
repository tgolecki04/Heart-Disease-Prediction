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
    print("Warning: TensorFlow not installed. Keras models cannot be loaded.")

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# --- Configuration ---

# Ścieżka do modelu. Może to być:
# 1. Plik .pkl (słownik z modelem i artefaktami)
# 2. Plik .pkl (sam model RF/XGB)
# 3. Plik .keras / .h5 (model sieci neuronowej)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/medical_heart_risk_model.keras")

# Opcjonalna ścieżka do artefaktów. Jeśli None, system spróbuje znaleźć je automatycznie.
ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", None)

# Feature order MUST match the order used during training
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
    "type": "unknown"  # sklearn, keras, torch
}


def try_load_external_artifacts(main_model_path: str):
    """
    Próbuje załadować artefakty (scaler, imputer) z zewnętrznego pliku.
    1. Sprawdza zmienną środowiskową ARTIFACTS_PATH.
    2. Jeśli pusta, szuka pliku '{nazwa_modelu}_artifacts.pkl'.
    """
    path_to_check = ARTIFACTS_PATH

    # Auto-discovery logic
    if not path_to_check:
        base_name = os.path.splitext(main_model_path)[0]
        path_to_check = f"{base_name}_artifacts.pkl"
        print(f"Auto-discovery: Checking for artifacts at {path_to_check}")

    if os.path.exists(path_to_check):
        try:
            data = joblib.load(path_to_check)
            if isinstance(data, dict):
                ARTIFACTS["scaler"] = data.get("scaler")
                ARTIFACTS["imputer"] = data.get("imputer")
                # Opcjonalnie nadpisujemy próg, jeśli jest w artefaktach
                if "optimal_threshold" in data:
                    ARTIFACTS["threshold"] = data.get("optimal_threshold")
                print(f"✅ Loaded auxiliary artifacts from: {path_to_check}")
            else:
                print(f"⚠️ Artifacts file found at {path_to_check} but is not a dictionary.")
        except Exception as e:
            print(f"⚠️ Failed to load artifacts from {path_to_check}: {e}")
    else:
        print(f"ℹ️ No auxiliary artifacts found at {path_to_check}. Assuming raw input or pipeline embedded in model.")


def determine_model_type(model):
    """Rozpoznaje typ załadowanego modelu."""
    if keras and isinstance(model, keras.Model):
        return "keras"
    if torch and isinstance(model, nn.Module):
        return "torch"
    # Sprawdzenie dla Scikit-Learn / XGBoost (Standard API)
    if hasattr(model, "predict") or hasattr(model, "predict_proba"):
        return "sklearn"
    return "unknown"


@app.on_event("startup")
def load_all_resources():
    print(f"🚀 Starting loading procedure for: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ CRITICAL: Model file not found at {MODEL_PATH}")
        return

    try:
        # --- Ścieżka 1: Model Keras (.keras / .h5) ---
        if MODEL_PATH.endswith(".keras") or MODEL_PATH.endswith(".h5"):
            if keras is None:
                raise ImportError("TensorFlow/Keras is required to load this model.")

            print("📦 Detected Keras format. Loading...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            ARTIFACTS["model"] = model
            ARTIFACTS["type"] = "keras"

            # Keras rzadko zawiera scaler w sobie, szukamy na zewnątrz
            try_load_external_artifacts(MODEL_PATH)

        # --- Ścieżka 2: Pickle (.pkl / .joblib) ---
        else:
            print("📦 Detected Pickle format. Loading via joblib...")
            data = joblib.load(MODEL_PATH)

            # SCENARIUSZ A: Plik to słownik (Pełny Pipeline: model + artefakty)
            if isinstance(data, dict) and "model" in data:
                print("🔹 Structure: Dictionary containing model and artifacts.")
                ARTIFACTS["model"] = data.get("model")
                ARTIFACTS["scaler"] = data.get("scaler")
                ARTIFACTS["imputer"] = data.get("imputer")
                ARTIFACTS["threshold"] = data.get("optimal_threshold", 0.5)

            # SCENARIUSZ B: Plik to sam Model (np. XGBClassifier, RandomForest)
            else:
                print("🔹 Structure: Raw Model Object.")
                ARTIFACTS["model"] = data
                # Skoro to sam model, musimy poszukać scalera/imputera obok
                try_load_external_artifacts(MODEL_PATH)

            # Wykrycie typu modelu (bo w pkl może być wszystko, nawet Keras wrapper)
            ARTIFACTS["type"] = determine_model_type(ARTIFACTS["model"])

        print(f"✅ Model Loaded Successfully.")
        print(f"   Type: {ARTIFACTS['type']}")
        print(f"   Scaler: {'Yes' if ARTIFACTS['scaler'] else 'No'}")
        print(f"   Imputer: {'Yes' if ARTIFACTS['imputer'] else 'No'}")
        print(f"   Threshold: {ARTIFACTS['threshold']}")

    except Exception as e:
        print(f"❌ CRITICAL LOAD ERROR: {e}")
        traceback.print_exc()


# python
@app.post("/predict")
def predict(req: PredictRequest):
    if ARTIFACTS["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # 1. Przygotowanie danych
        input_data = {k: float(getattr(req, k)) for k in EXPECTED_FEATURES}
        df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)

        model = ARTIFACTS["model"]
        m_type = ARTIFACTS["type"]
        prob = 0.0

        # Decide how to preprocess / what to pass to the model:
        # If model is a sklearn Pipeline or expects feature names, give it the DataFrame.
        use_dataframe_for_model = False
        try:
            # pipeline detection or estimator that stores feature names
            if hasattr(model, "named_steps") or hasattr(model, "feature_names_in_"):
                use_dataframe_for_model = True
        except Exception:
            use_dataframe_for_model = False

        # If model expects DataFrame, skip external imputer/scaler (assume pipeline handles it)
        if use_dataframe_for_model:
            x_for_model = df
        else:
            # Apply external preprocessing if provided
            x_processed = df.copy()
            if ARTIFACTS["imputer"]:
                x_processed = ARTIFACTS["imputer"].transform(x_processed)
            if ARTIFACTS["scaler"]:
                x_processed = ARTIFACTS["scaler"].transform(x_processed)
            x_for_model = x_processed

        # --- KERAS ---
        if m_type == "keras":
            if isinstance(x_for_model, pd.DataFrame):
                x_for_model = x_for_model.values
            raw = model.predict(x_for_model, verbose=0)
            if raw.shape[-1] == 1:
                prob = float(raw[0][0])
            else:
                prob = float(raw[0][1])

        # --- TORCH ---
        elif m_type == "torch":
            if isinstance(x_for_model, pd.DataFrame):
                x_for_model = x_for_model.values
            tensor_in = torch.tensor(x_for_model, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_in)
                val = logits.item()
                prob = 1 / (1 + np.exp(-val))

        # --- SKLEARN / XGBOOST (Standard API) ---
        else:
            if hasattr(model, "predict_proba"):
                # ensure correct input type for sklearn
                prob_arr = model.predict_proba(x_for_model)
                if prob_arr.shape[1] == 2:
                    prob = float(prob_arr[0][1])
                else:
                    prob = float(prob_arr[0][0])
            elif hasattr(model, "predict"):
                pred = model.predict(x_for_model)
                # pred może być tablicą lub wartościami skalarnymi
                prob = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            else:
                raise ValueError("Model does not support predict or predict_proba")

        threshold = ARTIFACTS["threshold"]
        is_risk = 1 if prob >= threshold else 0

        return {
            "class": is_risk,
            "probability": round(prob, 4),
            "threshold_used": threshold,
            "model_type": m_type
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": ARTIFACTS["model"] is not None,
        "model_type": ARTIFACTS["type"],
        "artifacts": {
            "scaler": ARTIFACTS["scaler"] is not None,
            "imputer": ARTIFACTS["imputer"] is not None
        }
    }