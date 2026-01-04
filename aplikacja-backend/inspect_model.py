#!/usr/bin/env python3

"""
Prosty inspektor plików modeli ML

Obsługuje:
 1. Pickle/Joblib (Standardowe modele Scikit-Learn/XGBoost)
 2. TensorFlow/Keras (wraz z automatycznym wykrywaniem plików towarzyszących _artifacts.pkl)
 3. PyTorch

Użycie:
  python inspect_model.py <path-to-file-or-dir>

Co robi:
 - próbuje załadować ścieżkę sekwencją loaderów:
    joblib/pickle, xgboost natywny, tensorflow.keras, torch
 - podsumowuje załadowany obiekt i szuka wewnątrz kontenerów (dict/list/tuple)
    kandydatów na modele pod popularnymi kluczami jak 'model', 'estimator', 'clf' itd.
 - wypisuje nazwę klasy i czy istnieją metody predict/predict_proba

Uwagi:
 - Skrypt jest defensywny: nie będzie ponownie podnosił wyjątków loadera (spróbuje innych loaderów)
 - Jeśli klasa modelu jest brakująca, używa SafeUnpickler, do zastąpienia brakujących klas obiektami Stub,
    aby umożliwić inspekcję struktury bez błędów
"""

import sys
import os
import pickle
import joblib
import warnings

# Importy warunkowe
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# Bezpieczny Unpickler (dla brakujących klas)
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError, AttributeError):
            return type(name, (object,), {
                "__module__": module,
                "__repr__": lambda _: f"<MISSING CLASS: {module}.{name}>"
            })


# Loadery
def load_keras_model(path):
    if not tf:
        return None, "TensorFlow not installed"
    try:
        # compile=False pozwala załadować model nawet bez definicji custom loss function
        model = tf.keras.models.load_model(path, compile=False)
        return model, "tf.keras.models.load_model"
    except (ImportError, IOError, ValueError, TypeError) as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def load_torch_model(path):
    if not torch:
        return None, "PyTorch not installed"
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
        return obj, "torch.load"
    except (RuntimeError, IOError, pickle.UnpicklingError) as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def safe_pickle_load(path):
    load_errors = (OSError, IOError, pickle.UnpicklingError, AttributeError, EOFError, TypeError, RuntimeError)

    # Najpierw joblib (standard)
    try:
        return joblib.load(path), "joblib"
    except load_errors:
        pass

    # Potem SafeUnpickler (jeśli brakuje klas w kodzie)
    try:
        with open(path, "rb") as f:
            return SafeUnpickler(f).load(), "SafeUnpickler (Mocked Classes)"
    except load_errors:
        pass

    return None, "Failed all pickle methods"


# Logika Sidecar (Artefakty obok modelu)
def inspect_sidecar_artifacts(model_path):
    # Dla modeli Keras/Torch sprawdza, czy obok istnieje plik z artefaktami (scalerem)
    # Konwencja: {nazwa_modelu}_artifacts.pkl
    base_name = os.path.splitext(model_path)[0]
    # Lista potencjalnych nazw plików z artefaktami
    candidates = [
        f"{base_name}_artifacts.pkl",
        f"{base_name}_artifacts.joblib",
        os.path.join(os.path.dirname(model_path), "artifacts.pkl")
    ]

    found_path = None
    for c in candidates:
        if os.path.exists(c):
            found_path = c
            break

    if found_path:
        print(f"\nDetected Companion Artifacts File: {found_path}")
        print("-" * 40)
        obj, method = safe_pickle_load(found_path)
        if obj is not None:
            analyze(obj, name="Artifacts")
        else:
            print("Failed to load artifacts file.")
    else:
        print("\nNo companion artifacts file found (checked: *_artifacts.pkl).")
        print("If this is a Neural Network, make sure scaling is handled inside the model or externally.")


# Analiza
def analyze(obj, depth=0, name="Root"):
    indent = "  " * depth
    prefix = f"{indent}- {name}: "

    if depth > 10:
        print(f"{prefix} ... (max depth reached)")
        return

    obj_type = type(obj).__name__

    # Słowniki
    if isinstance(obj, dict):
        print(f"{prefix} Dict keys: {list(obj.keys())}")
        for k, v in obj.items():
            analyze(v, depth + 1, name=f"['{k}']")

    # Listy i Krotki
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{obj_type} (len={len(obj)})")
        for i, item in enumerate(obj):
            if i >= 5:
                print(f"{indent} ... ({len(obj) - 5} more items)")
                break
            analyze(item, depth + 1, name=f"[{i}]")

    # Brakujące obiekty (Stub)
    elif "MISSING CLASS" in repr(obj):
        print(f"{prefix}!  {repr(obj)}")
        print(f"{indent}   -> Main app will fail unless this class is imported!")

    # Keras Model
    elif tf and isinstance(obj, tf.keras.Model):
        print(f"{prefix} Keras Model")
        try:
            input_shape = getattr(obj, "input_shape", "N/A")
            print(f"{indent} - Input Shape: {input_shape}")

            # Podstawowe informacje o warstwach
            layers = [l.__class__.__name__ for l in obj.layers]
            if len(layers) > 5:
                print(f"{indent} - Layers: {layers[:5]} + {len(layers) - 5} more...")
            else:
                print(f"{indent} - Layers: {layers}")
        except (AttributeError, ValueError):
            pass

    # PyTorch Model
    elif torch and isinstance(obj, nn.Module):
        print(f"{prefix} PyTorch nn.Module")
        print(f"{indent} Params: {sum(p.numel() for p in obj.parameters())}")

    # Inne obiekty (Sklearn itp.)
    else:
        info = []
        if hasattr(obj, "predict"): info.append("predict")
        if hasattr(obj, "transform"): info.append("transform")
        if hasattr(obj, "data_min_"): info.append("fitted_scaler")  # np. MinMaxScaler
        if hasattr(obj, "mean_"): info.append("fitted_scaler")  # np. StandardScaler

        extra = f" [{', '.join(info)}]" if info else ""
        print(f"{prefix}{obj_type}{extra}")

        # Jeśli to Pipeline, wejdź głębiej
        if hasattr(obj, "steps"):
            print(f"{indent}   Pipeline Steps:")
            try:
                for step_name, step_obj in obj.steps:
                    analyze(step_obj, depth + 2, name=f"Step: {step_name}")
            except (ValueError, TypeError):
                print(f"{indent}   (Could not iterate over steps)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        sys.exit(1)

    print(f"Inspecting: {path}")
    print("-" * 40)

    obj = None
    method = "unknown"
    is_keras_or_torch = False

    # 1. Próba Keras
    if path.endswith((".keras", ".h5")):
        obj, method = load_keras_model(path)
        is_keras_or_torch = True

    # 2. Próba PyTorch
    elif path.endswith((".pt", ".pth")):
        obj, method = load_torch_model(path)
        is_keras_or_torch = True

    # 3. Próba Pickle/Joblib
    if obj is None:
        obj, m = safe_pickle_load(path)
        if obj is not None:
            method = m

    # 4. Fallback (np. keras bez rozszerzenia)
    if obj is None and not is_keras_or_torch:
        k_obj, k_method = load_keras_model(path)
        if k_obj:
            obj, method = k_obj, k_method
            is_keras_or_torch = True

    if obj is None:
        print("CRITICAL: Could not load file.")
        sys.exit(1)

    print(f"Loaded via: {method}")
    print("-" * 40)

    # Analiza głównego obiektu
    analyze(obj)

    # Jeśli to był model Keras/Torch, sprawdź, czy obok są artefakty (.pkl)
    if is_keras_or_torch:
        inspect_sidecar_artifacts(path)

    print("-" * 40)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()