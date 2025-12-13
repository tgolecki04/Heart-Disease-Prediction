#!/usr/bin/env python3
"""
Szybki skrypt diagnostyczny do sprawdzenia zawartości pliku modelu (joblib/pickle/xgboost).
Użycie:
  python backend/inspect_model.py backend/model/model.pkl
"""
import sys
import os
import pickle
import joblib
import xgboost as xgb
import traceback

def try_load(path):
    if not os.path.exists(path):
        print("Plik nie istnieje:", path)
        sys.exit(2)
    # joblib
    try:
        m = joblib.load(path)
        return m, "joblib"
    except Exception:
        pass
    # pickle
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        return m, "pickle"
    except Exception:
        pass
    # xgboost native
    try:
        booster = xgb.Booster()
        booster.load_model(path)
        return booster, "xgboost.Booster"
    except Exception:
        pass
    return None, None

def summary(obj, prefix=""):
    t = type(obj)
    print(f"{prefix}type: {t}")
    # basic introspection for containers / common cases
    if isinstance(obj, dict):
        print(f"{prefix}dict keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{prefix} key -> {k} : {type(v)}; has_predict={hasattr(v, 'predict')}; has_predict_proba={hasattr(v, 'predict_proba')}; is_booster={isinstance(v, xgb.Booster)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{t} len={len(obj)}")
        for i, v in enumerate(obj[:10]):
            print(f"{prefix} [{i}] -> {type(v)}; has_predict={hasattr(v, 'predict')}; has_predict_proba={hasattr(v, 'predict_proba')}; is_booster={isinstance(v, xgb.Booster)}")
    else:
        print(f"{prefix}has predict: {hasattr(obj, 'predict')}; has predict_proba: {hasattr(obj, 'predict_proba')}; is xgboost.Booster: {isinstance(obj, xgb.Booster)}")
        # optionally print repr (truncated)
        try:
            r = repr(obj)
            if len(r) > 400:
                r = r[:400] + " ... (truncated)"
            print(f"{prefix}repr: {r}")
        except Exception:
            pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path-to-model>")
        sys.exit(1)
    path = sys.argv[1]
    obj, how = try_load(path)
    if obj is None:
        print("Nie udało się załadować pliku jako joblib/pickle/xgboost.Booster.")
        sys.exit(3)
    print("Załadowano metodą:", how)
    summary(obj, prefix="  ")
    # If dict or container try to find candidate model
    candidates = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "predict") or hasattr(v, "predict_proba") or isinstance(v, xgb.Booster):
                candidates.append((k, v))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if hasattr(v, "predict") or hasattr(v, "predict_proba") or isinstance(v, xgb.Booster):
                candidates.append((i, v))
    if candidates:
        print("\nZnaleziono potencjalne modele wewnątrz kontenera:")
        for k, v in candidates:
            print("  ->", k, "type:", type(v), "has_predict:", hasattr(v, "predict"), "is_booster:", isinstance(v, xgb.Booster))
    else:
        print("\nNie znalazłem żadnego obiektu z metodą predict/predict_proba ani xgboost.Booster wewnątrz. Możliwe, że plik zawiera sam słownik/metadane.")
    print("\nKoniec diagnostyki.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(10)