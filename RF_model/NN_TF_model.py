import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import ADASYN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Wczytanie i przygotowanie danych
data = pd.read_csv("framingham_heart_study.csv")
if 'education' in data.columns:
    data = data.drop(columns=["education"])

feature_names = data.drop(columns=["TenYearCHD"], axis=1).columns.tolist()
X = data.drop(columns=["TenYearCHD"], axis=1)
y = data["TenYearCHD"]

# 2. Uzupełnianie brakujących wartości i skalowanie
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Podział na zbiory z zachowaniem proporcji klas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# 4. Balansowanie klas (ADASYN)
sm = ADASYN(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 5. OPTYMALNA ARCHITEKTURA - NAJLEPSZE PARAMETRY
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_res.shape[1],)),

    # Warstwa 1 - optymalna regularyzacja
    tf.keras.layers.Dense(64, activation="elu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    # Warstwa 2
    tf.keras.layers.Dense(32, activation="elu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    # Warstwa wyjściowa
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# 6. FUNKCJA STRATY Z OPTYMALNĄ WAGĄ DLA MEDYCYNY
def medical_loss(y_true, y_pred, fn_weight=6.0):
    """Custom loss function that penalizes false negatives for medical safety"""
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    fn = y_true * (1 - y_pred)  # False negatives
    return bce + fn_weight * tf.reduce_mean(fn)


# 7. KOMPILACJA Z OPTYMALNYMI PARAMETRAMI
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=medical_loss,
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.Precision(name='precision')]
)

# 8. Przygotowanie danych walidacyjnych
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_res, y_train_res, test_size=0.2, random_state=42, stratify=y_train_res
)

# 9. CALLBACKS - OPTYMALNE USTAWIENIA
early_stop = EarlyStopping(
    monitor='val_recall',
    patience=15,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

# 10. TRENOWANIE MODELU
print("🚀 Trenowanie modelu...")
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=100,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 11. PREDYKCJE Z OPTYMALNYM PROGIEM DLA MEDYCYNY
y_pred_prob = model.predict(X_test, verbose=0).flatten()

# REKOMENDOWANY PRÓG MEDYCZNY: 0.6 (dla bezpieczeństwa pacjentów)
OPTIMAL_THRESHOLD = 0.6
y_pred = (y_pred_prob >= OPTIMAL_THRESHOLD).astype(int)

# 12. PODSTAWOWA OCENA
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n{'=' * 60}")
print("PODSTAWOWA OCENA MODELU")
print(f"{'=' * 60}")
print(f"Próg klasyfikacji: {OPTIMAL_THRESHOLD}")
print(f"\nMacierz pomyłek:")
print(f"TN: {tn:4d} | FP: {fp:4d}")
print(f"FN: {fn:4d} | TP: {tp:4d}")
print(f"\nMetryki:")
print(f"Czułość (Recall):     {recall:.3f}")
print(f"Precyzja:             {precision:.3f}")
print(f"Specyficzność:        {specificity:.3f}")
print(f"\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=['Niskie ryzyko', 'Wysokie ryzyko']))

# 13: Zapisywanie modelu (Format keras z custom loss)
# python
import os
import joblib
import tensorflow as tf

print("\nZapis modelu i artefaktów")

# Ścieżki plików
MODEL_PATH = os.path.join(os.getcwd(), "medical_heart_risk_model.keras")
ARTIFACTS_PATH = os.path.join(os.getcwd(), "medical_heart_risk_model_artifacts.pkl")

# Odtwórz model z konfiguracji (usuwa odniesienia do custom loss/optimizer)
inference_model = tf.keras.Sequential.from_config(model.get_config())
inference_model.set_weights(model.get_weights())

# Zapis modelu gotowego do inferencji (usuń parametr save_traces)
try:
    inference_model.save(MODEL_PATH, include_optimizer=False)
    print(f"Model zapisany do: {MODEL_PATH}")
except Exception as e:
    print(f"Błąd przy zapisie modelu: {e}")
    raise

# Zapis preprocessing i metadanych w jednym pliku .pkl
artifacts = {
    "imputer": imputer,
    "scaler": scaler,
    "feature_names": feature_names,
    "optimal_threshold": float(OPTIMAL_THRESHOLD),
    "note": "Preprocessing i metadane do użycia z modelem .keras"
}

try:
    joblib.dump(artifacts, ARTIFACTS_PATH, compress=3)
    print(f"Artefakty zapisane do: {ARTIFACTS_PATH}")
except Exception as e:
    print(f"Błąd przy zapisie artefaktów: {e}")
    raise

print("\nPrzykład ładowania w API:")
print(f"1) model = tf.keras.models.load_model(r'{MODEL_PATH}', compile=False)")
print(f"2) artifacts = joblib.load(r'{ARTIFACTS_PATH}')")
print("   scaler = artifacts['scaler']; imputer = artifacts['imputer']")
print("   feature_names = artifacts['feature_names']; threshold = artifacts['optimal_threshold']")

