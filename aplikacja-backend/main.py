from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import pandas as pd
import traceback
import math
from typing import Dict, Any

# Importy warunkowe dla TensorFlow i PyTorch
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

# Konfiguracja ścieżek
# Ścieżką może być:
# 1. Plik .pkl (słownik z modelem i artefaktami)
# 2. Plik .pkl (sam model RF/XGB)
# 3. Plik .keras/.h5 (model sieci neuronowej)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/rf_baseline2.pkl")

# Opcjonalna ścieżka do artefaktów, jeśli None, system spróbuje znaleźć je automatycznie
ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", None)

# Oczekiwane cechy wejściowe modelu (w kolejności zgodnej z treningiem)
EXPECTED_FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

# Globalne przechowywanie załadowanych artefaktów
ARTIFACTS: Dict[str, Any] = {
    "model": None,
    "scaler": None,
    "imputer": None,
    "threshold": 0.5,
    "type": "unknown"  # sklearn, keras, torch
}

# Funkcje pomocnicze

def try_load_external_artifacts(main_model_path: str):
    # Próbuje załadować artefakty (scaler, imputer) z zewnętrznego pliku
    # 1. Sprawdza zmienną środowiskową ARTIFACTS_PATH
    # 2. Jeśli pusta, szuka pliku '{nazwa_modelu}_artifacts.pkl'
    path_to_check = ARTIFACTS_PATH

    # Auto-discovery, jeśli nie podano ścieżki
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
                print(f"Loaded auxiliary artifacts from: {path_to_check}")
            else:
                print(f"! Artifacts file found at {path_to_check} but is not a dictionary.")
        except Exception as e:
            print(f"! Failed to load artifacts from {path_to_check}: {e}")
    else:
        print(f"No auxiliary artifacts found. Assuming raw input or pipeline embedded in model.")


def determine_model_type(model: Any) -> str:
    # Rozpoznanie typu załadowanego modelu
    if keras and isinstance(model, keras.Model):
        return "keras"
    if torch and isinstance(model, nn.Module):
        return "torch"
    # Sprawdzenie dla Scikit-Learn/XGBoost (Standard API)
    if hasattr(model, "predict") or hasattr(model, "predict_proba"):
        return "sklearn"
    return "unknown"


def load_model_logic():
    # Główna logika ładowania modeli (wywoływana przy starcie)
    print(f"Starting loading procedure for: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model file not found at {MODEL_PATH}")
        return

    try:
        # 1. Keras (.keras/.h5)
        if MODEL_PATH.endswith(".keras") or MODEL_PATH.endswith(".h5"):
            if keras is None:
                raise ImportError("TensorFlow/Keras is required to load this model.")

            print("Detected Keras format. Loading with compile=False...")
            # Opcja compile=False zapobiega błędom o brakujących funkcjach straty (np. medical_loss)
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            ARTIFACTS["model"] = model
            ARTIFACTS["type"] = "keras"
            # Keras rzadko zawiera scaler w sobie, szukamy na zewnątrz
            try_load_external_artifacts(MODEL_PATH)

        # 2. Pickle/Joblib
        else:
            print("Detected Pickle format. Loading via joblib...")
            data = joblib.load(MODEL_PATH)

            # A: Plik to słownik (Pełny Pipeline: model + artefakty)
            if isinstance(data, dict) and "model" in data:
                print("Structure: Dictionary (Full Pipeline).")
                ARTIFACTS["model"] = data.get("model")
                ARTIFACTS["scaler"] = data.get("scaler")
                ARTIFACTS["imputer"] = data.get("imputer")
                ARTIFACTS["threshold"] = data.get("optimal_threshold", 0.5)
            # B: Plik to sam Model (np. XGBClassifier, RandomForest)
            else:
                print("Structure: Raw Model Object.")
                ARTIFACTS["model"] = data
                try_load_external_artifacts(MODEL_PATH)

            # Wykrycie typu modelu (bo w pkl może być wszystko, nawet Keras wrapper)
            ARTIFACTS["type"] = determine_model_type(ARTIFACTS["model"])

        print(f"Model Loaded Successfully.")
        print(f"    Type: {ARTIFACTS['type']}")
        print(f"    Scaler: {'Yes' if ARTIFACTS['scaler'] else 'No'}")
        print(f"   Imputer: {'Yes' if ARTIFACTS['imputer'] else 'No'}")
        print(f"   Threshold: {ARTIFACTS['threshold']}")

    except Exception as e:
        print(f"CRITICAL LOAD ERROR: {e}")
        traceback.print_exc()

# Lifespan Manager
@asynccontextmanager
async def lifespan(_: FastAPI):
    # Logika startowa
    load_model_logic()
    yield
    # Logika zamykania (opcjonalnie)
    ARTIFACTS.clear()
    print("Application shutdown. Artifacts cleared.")

# Inicjalizacja Aplikacji
app = FastAPI(title="Medical Heart Risk API", lifespan=lifespan)

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

@app.post("/predict")
def predict(req: PredictRequest):
    model = ARTIFACTS["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        # 1. Przygotowanie danych
        input_data = {k: float(getattr(req, k)) for k in EXPECTED_FEATURES}
        df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)

        # Lokalne referencje dla lepszego typowania i wydajności
        scaler = ARTIFACTS["scaler"]
        imputer = ARTIFACTS["imputer"]
        m_type = ARTIFACTS["type"]
        threshold = ARTIFACTS["threshold"]

        # 2. Preprocessing
        # Decyzja: czy model (np. Pipeline) obsługuje transformacje sam, czy robimy to ręcznie?
        # Zakładamy ręczne, chyba że to surowy sklearn pipeline, ale dla bezpieczeństwa,
        # jeśli mamy załadowany scaler w ARTIFACTS, używamy go.

        x_processed = df.copy()

        if imputer is not None:
            # Wywołanie eliminuje ostrzeżenie "Cannot find reference"
            x_processed = imputer.transform(x_processed)

        if scaler is not None:
            x_processed = scaler.transform(x_processed)

        # Konwersja do formatu akceptowalnego przez model
        if m_type in ["keras", "torch"] and isinstance(x_processed, pd.DataFrame):
            x_final = x_processed.values
        else:
            x_final = x_processed

        # 3. Predykcja
        prob = 0.0

        if m_type == "keras":
            # Keras
            raw = model.predict(x_final, verbose=0)

            # Obsługa binarnej klasyfikacji
            if raw.shape[-1] == 1:
                val = float(raw[0][0])
                # Detekcja logitów
                if val < 0 or val > 1:
                    prob = 1 / (1 + math.exp(-val))
                else:
                    prob = val
            else:
                prob = float(raw[0][1])

        elif m_type == "torch":
            # Torch
            if torch:
                tensor_in = torch.tensor(x_final, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(tensor_in)
                    val = logits.item()
                    prob = 1 / (1 + math.exp(-val))

        else:
            # Sklearn/XGBoost
            if hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(x_final)
                if prob_arr.shape[1] == 2:
                    prob = float(prob_arr[0][1])
                else:
                    prob = float(prob_arr[0][0])
            elif hasattr(model, "predict"):
                pred = model.predict(x_final)
                prob = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            else:
                raise ValueError("Model object has no predict/predict_proba method")

        # 4. Wynik
        is_risk = 1 if prob >= threshold else 0

        return {
            "class": is_risk,
            "probability": round(prob, 4),
            "threshold_used": threshold,
            "model_type": m_type
        }

    except Exception as e:
        # Łapiemy tylko błędy podczas samej predykcji/transformacji
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction processing failed: {str(e)}")


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