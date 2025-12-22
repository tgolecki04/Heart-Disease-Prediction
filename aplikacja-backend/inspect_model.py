#!/usr/bin/env python3

"""
Inspect model helper.

Usage:
  python inspect_model.py <path-to-file-or-dir>

What it does:
 - prints versions of key libraries (if zainstalowane)
 - tries to load the path with a sequence of loaders:
     joblib / pickle, xgboost native, tensorflow.keras, torch
 - summarizes the loaded object and searches inside containers (dict/list/tuple)
   for candidate models under common keys like 'model', 'estimator', 'clf' etc.
 - for Keras models prints summary, for PyTorch modules prints parameter count,
   for sklearn-like estimators prints class name and whether predict/predict_proba exist.

Notes:
 - If your pickle contains complex objects, tensorflow/torch may be required
   in the environment to fully inspect and resave them.
 - The script is defensive: it won't re-raise loader exceptions (it will try other loaders).
"""

"""
Inspect model helper (Robust / Safe Mode).
Can load pickles even if the original model class definition is missing.
"""

import sys
import os
import pickle
import joblib
import warnings
import io

# --- 0. Optional Imports ---
try:
    import tensorflow as tf
except ImportError:
    tf = None

# --- 1. Safe Unpickler (The Magic Fix) ---
class SafeUnpickler(pickle.Unpickler):
    """
    A custom unpickler that doesn't crash if a class is missing.
    It replaces missing classes with a simple 'Stub' object.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            # Create a dummy class on the fly
            # print(f"Warning: Could not find class '{module}.{name}'. Creating Stub.")
            return type(name, (object,), {
                "__module__": module,
                "__repr__": lambda self: f"<Stub for {module}.{name}>"
            })


def safe_pickle_load(path):
    """
    Attempts to load via joblib first. If that fails (due to missing classes),
    falls back to SafeUnpickler.
    """
    try:
        # Standard load
        return joblib.load(path), "joblib"
    except Exception as e_joblib:
        # Fallback: Read file bytes and use SafeUnpickler
        try:
            with open(path, "rb") as f:
                return SafeUnpickler(f).load(), "SafeUnpickler (Mocked)"
        except Exception as e_safe:
            return None, f"Failed both joblib ({str(e_joblib)}) and SafeUnpickler ({str(e_safe)})"

def load_keras_model(path):
    """Loads a model from a .keras file."""
    if not tf:
        return None, "TensorFlow is not installed, cannot load .keras file."
    try:
        # We load without custom_objects, as we don't know them in this script.
        # This is usually sufficient for inspecting the architecture.
        model = tf.keras.models.load_model(path, compile=False)
        return model, "tensorflow.keras.models.load_model"
    except Exception as e:
        return None, f"Failed to load with TensorFlow/Keras: {str(e)}"


def analyze(obj, depth=0, name="Root"):
    indent = "  " * depth
    prefix = f"{indent}- {name}: "

    obj_type = type(obj).__name__

    # 1. Handle Dictionaries (Recurse)
    if isinstance(obj, dict):
        print(f"{prefix}Dict with keys: {list(obj.keys())}")
        for k, v in obj.items():
            analyze(v, depth + 1, name=f"['{k}']")

    # 2. Handle Lists (Summarize)
    elif isinstance(obj, list):
        print(f"{prefix}List (len={len(obj)})")

    # 3. Handle Stub Objects (Missing Classes)
    elif "Stub" in repr(obj):
        print(f"{prefix}⚠️  MISSING CLASS: {repr(obj)}")
        print(f"{indent}   (This means 'main.py' needs this class defined to run predictions)")

    # 4. Handle Standard Models
    else:
        info = ""
        is_keras = "keras" in str(type(obj)).lower()

        if hasattr(obj, "predict"): info += " [Has predict]"
        if hasattr(obj, "predict_proba"): info += " [Has predict_proba]"
        if hasattr(obj, "transform"): info += " [Has transform (Scaler/Imputer)]"

        # Check for PyTorch/Keras
        if is_keras: info += " [Keras Model]"
        if "torch" in str(type(obj)).lower(): info += " [PyTorch Model]"

        print(f"{prefix}{obj_type}{info}")

        if is_keras and hasattr(obj, 'summary'):
            # Capture summary() output to control indentation
            stream = io.StringIO()
            # Redirect stdout to the stream
            sys.stdout = stream
            obj.summary()
            # Restore original stdout
            sys.stdout = sys.__stdout__
            summary_string = stream.getvalue()
            indented_summary = "\n".join([f"{indent}    {line}" for line in summary_string.split("\n")])
            print(f"{indent}  Model Summary:\n{indented_summary}")



def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_pkl>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print("File not found.")
        sys.exit(1)

    print(f"--- Inspecting: {path} ---")

    obj, method = (None, None)

    if path.endswith(".keras"):
        obj, method = load_keras_model(path)
    else:
        # Assume pickle-based format for any other extension
        obj, method = safe_pickle_load(path)

    if obj is None:
        print(f"CRITICAL ERROR: {method}")
        sys.exit(1)

    print(f"Loaded successfully using: {method}")
    print("--- Structure Analysis ---")
    analyze(obj)
    print("--- End of Analysis ---")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()