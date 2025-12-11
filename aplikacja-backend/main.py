from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import traceback
import types

# Config: path to model file (default relative to backend/)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")

# Feature order must match training
FEATURE_NAMES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

app = FastAPI(title="Heart Disease Prediction API")

# CORS - allow all for local testing. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    male: float = Field(..., example=1)
    age: float = Field(..., example=54)
    currentSmoker: float = Field(..., example=0)
    cigsPerDay: float = Field(..., example=0)
    BPMeds: float = Field(..., example=0)
    prevalentStroke: float = Field(..., example=0)
    prevalentHyp: float = Field(..., example=0)
    diabetes: float = Field(..., example=0)
    totChol: float = Field(..., example=213)
    sysBP: float = Field(..., example=132)
    diaBP: float = Field(..., example=85)
    BMI: float = Field(..., example=26.2)
    heartRate: float = Field(..., example=80)
    glucose: float = Field(..., example=83)

MODEL = {"obj": None, "is_booster": False, "has_proba": False, "raw_loaded": None}


def _is_model_like(o):
    return (hasattr(o, "predict") and callable(getattr(o, "predict"))) or isinstance(o, xgb.Booster) or (hasattr(o, "predict_proba") and callable(getattr(o, "predict_proba")))


def _extract_model_from_container(container):
    """
    Jeśli container to dict/list/tuple zawierający model, spróbuj go znaleźć.
    Zwraca (model_obj, reason) albo (None, None) jeśli nie znaleziono.
    """
    # common keys to check first
    if isinstance(container, dict):
        for key in ("model", "estimator", "clf", "classifier", "pipeline", "xgb_model", "xgb"):
            if key in container:
                candidate = container[key]
                if _is_model_like(candidate):
                    return candidate, f"found in dict key '{key}'"
        # otherwise scan values
        for k, v in container.items():
            if _is_model_like(v):
                return v, f"found in dict value under key '{k}'"
    elif isinstance(container, (list, tuple)):
        for i, v in enumerate(container):
            if _is_model_like(v):
                return v, f"found in list/tuple index {i}"
    return None, None


def try_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    # 1) try joblib
    try:
        m = joblib.load(path)
        return m, "joblib"
    except Exception:
        pass

    # 2) try pickle
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        return m, "pickle"
    except Exception:
        pass

    # 3) try xgboost native Booster
    try:
        booster = xgb.Booster()
        booster.load_model(path)
        return booster, "xgboost.Booster"
    except Exception:
        pass

    raise RuntimeError("Could not load model with joblib/pickle or xgboost.Booster")


@app.on_event("startup")
def load_model_on_startup():
    try:
        raw_obj, method = try_load_model(MODEL_PATH)
        MODEL["raw_loaded"] = {"type": type(raw_obj), "loader_method": method}
        # If raw_obj itself looks like a model, use it
        if _is_model_like(raw_obj):
            MODEL["obj"] = raw_obj
            MODEL["is_booster"] = isinstance(raw_obj, xgb.Booster)
            MODEL["has_proba"] = hasattr(raw_obj, "predict_proba") and callable(getattr(raw_obj, "predict_proba"))
            print(f"Loaded model directly (method={method}). type={type(raw_obj)}; is_booster={MODEL['is_booster']}; has_proba={MODEL['has_proba']}")
            return

        # If raw_obj is a container try to extract
        extracted, reason = _extract_model_from_container(raw_obj)
        if extracted is not None:
            MODEL["obj"] = extracted
            MODEL["is_booster"] = isinstance(extracted, xgb.Booster)
            MODEL["has_proba"] = hasattr(extracted, "predict_proba") and callable(getattr(extracted, "predict_proba"))
            print(f"Extracted model from container ({reason}). type={type(extracted)}; is_booster={MODEL['is_booster']}; has_proba={MODEL['has_proba']}")
            return

        # if we get here, we couldn't find a model-like object
        print("Loaded object is not a model-like object. Type:", type(raw_obj))
        # give helpful diagnostic printout
        try:
            if isinstance(raw_obj, dict):
                print("dict keys:", list(raw_obj.keys()))
            elif isinstance(raw_obj, (list, tuple)):
                print("container length:", len(raw_obj))
        except Exception:
            pass
        raise RuntimeError("Załadowany plik nie zawiera obiektu z metodą predict/predict_proba ani xgboost.Booster. Uruchom inspect_model.py aby zobaczyć zawartość pliku.")
    except Exception as e:
        print("Error loading model at startup:", e)
        traceback.print_exc()
        # Re-raise so startup fails and wyraźnie widzisz problem
        raise


@app.post("/predict")
def predict(req: PredictRequest):
    # Build a DataFrame with the correct feature order
    try:
        values = {k: float(getattr(req, k)) for k in FEATURE_NAMES}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input fields or types: {e}")

    df = pd.DataFrame([values], columns=FEATURE_NAMES)

    try:
        if MODEL["obj"] is None:
            # As fallback, try to inspect raw_loaded if possible and give user hint
            detail = "Model nie został poprawnie załadowany. Sprawdź logi serwera i uruchom backend/inspect_model.py aby zobaczyć zawartość pliku modelu."
            raise HTTPException(status_code=500, detail=detail)

        m = MODEL["obj"]

        # xgboost.Booster handling
        if isinstance(m, xgb.Booster):
            dmatrix = xgb.DMatrix(df)
            preds = m.predict(dmatrix)
            # preds najczęściej będą 1D probabilities
            if getattr(preds, "ndim", 1) == 1:
                prob = float(preds[0])
                pred_class = int(prob >= 0.5)
            else:
                prob = float(preds[0][1])
                pred_class = int(prob >= 0.5)
            return {"probability": prob, "class": pred_class}

        # If object is a dict or other container that slipped through, try to extract again
        if isinstance(m, dict) or isinstance(m, (list, tuple)):
            extracted, reason = _extract_model_from_container(m)
            if extracted:
                m = extracted
            else:
                raise HTTPException(status_code=500, detail="Załadowany obiekt modelu jest kontenerem (dict/list) bez wewnętrznego obiektu z metodą predict. Uruchom backend/inspect_model.py aby zobaczyć strukturę pliku i ponownie zapisz model jako sam model (joblib.dump(model, ...)).")

        # Now expect m to have sklearn-like API
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(df)[0][1]
            pred_class = int(m.predict(df)[0])
            return {"probability": float(proba), "class": pred_class}
        elif hasattr(m, "predict"):
            cls = m.predict(df)[0]
            try:
                cls_int = int(cls)
            except Exception:
                cls_int = None
            return {"probability": None, "class": cls_int}
        else:
            raise HTTPException(status_code=500, detail=f"Załadowany obiekt typu {type(m)} nie ma metod predict/predict_proba ani nie jest xgboost.Booster. Uruchom inspect_model.py aby zdiagnozować plik.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL["obj"] is not None, "raw_loaded": MODEL.get("raw_loaded")}