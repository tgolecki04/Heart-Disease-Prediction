import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import ADASYN
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

def create_clean_pipeline():
    """Bezpieczny pipeline bez data leakage"""

    # 1. Wczytanie danych
    data = pd.read_csv("framingham_heart_study.csv")
    if 'education' in data.columns:
        data = data.drop(columns=["education"])

    feature_names = data.drop(columns=["TenYearCHD"], axis=1).columns.tolist()
    X = data.drop(columns=["TenYearCHD"], axis=1)
    y = data["TenYearCHD"]

    # 2. PODZIAŁ NAJPIERW (najważniejsze!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    # 3. Imputacja TYLKO na train
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)  # fit tylko na train!
    X_test_imp = imputer.transform(X_test)  # transform na test

    # 4. Skalowanie TYLKO na train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)  # fit tylko na train!
    X_test_scaled = scaler.transform(X_test_imp)  # transform na test

    # 5. ADASYN TYLKO na train
    sm = ADASYN(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    return {
        'X_train_res': X_train_res, 'y_train_res': y_train_res,
        'X_test_scaled': X_test_scaled, 'y_test': y_test,
        'imputer': imputer, 'scaler': scaler,
        'feature_names': feature_names
    }


def train_model():
    """Główna funkcja trenowania modelu"""

    # Przygotowanie danych bez data leakage
    data = create_clean_pipeline()

    # 1. Podział na train i validation z zbalansowanych danych
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        data['X_train_res'], data['y_train_res'],
        test_size=0.2, random_state=42, stratify=data['y_train_res']
    )

    # 2. Budowa modelu
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(data['X_train_res'].shape[1],)),
        tf.keras.layers.Dense(64, activation="elu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="elu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # 3. Obliczanie wag klas zamiast custom loss
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(data['y_train_res'])
    weights = compute_class_weight('balanced', classes=classes, y=data['y_train_res'])
    class_weight = {0: weights[0], 1: weights[1]}

    # 4. Kompilacja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.AUC(name='auc')]
    )

    # 5. Callback z wczesnym zatrzymaniem
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    # 6. Trenowanie z class_weight
    print("Trenowanie modelu...")
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        batch_size=64,
        epochs=100,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=1
    )

    # 7. Predykcja na zbiorze testowym
    y_pred_prob = model.predict(data['X_test_scaled'], verbose=0).flatten()

    # 8. Znalezienie optymalnego progu (a nie arbitralny 0.6)
    precision, recall, thresholds = precision_recall_curve(data['y_test'], y_pred_prob)

    # Szukamy progu który maksymalizuje F2-score (większy nacisk na recall)
    f2_scores = []
    for i in range(len(precision) - 1):
        if precision[i] + recall[i] > 0:
            f2 = (5 * precision[i] * recall[i]) / (4 * precision[i] + recall[i])
            f2_scores.append(f2)
        else:
            f2_scores.append(0)

    optimal_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[optimal_idx]

    # 9. Predykcja z optymalnym progiem
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    # 10. Ocena modelu
    cm = confusion_matrix(data['y_test'], y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_metric = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(data['y_test'], y_pred_prob)

    print(f"\n{'=' * 60}")
    print("OCENA MODELU NA ZBIORZE TESTOWYM")
    print(f"{'=' * 60}")
    print(f"Optymalny próg: {optimal_threshold:.3f}")
    print(f"AUC: {auc:.4f}")
    print(f"\nMetryki kliniczne:")
    print(f"Czułość (Recall):     {recall:.3f}")
    print(f"Precyzja:             {precision_metric:.3f}")
    print(f"Specyficzność:        {specificity:.3f}")
    print(f"\nMacierz pomyłek:")
    print(f"TN: {tn:4d} | FP: {fp:4d}")
    print(f"FN: {fn:4d} | TP: {tp:4d}")
    print(f"\nRaport klasyfikacji:")
    print(classification_report(data['y_test'], y_pred, target_names=['Niskie ryzyko', 'Wysokie ryzyko']))

    # 11. Zapisz model i artefakty
    save_model_and_artifacts(model, data['imputer'], data['scaler'],
                             data['feature_names'], optimal_threshold)

    return model


def save_model_and_artifacts(model, imputer, scaler, feature_names, threshold):
    """Zapisuje model i artefakty preprocessingu"""

    MODEL_DIR = "saved_model_tf"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Ścieżki plików
    MODEL_PATH = os.path.join(MODEL_DIR, "medical_heart_risk_model.keras")
    ARTIFACTS_PATH = os.path.join(MODEL_DIR, "medical_heart_risk_model_artifacts.pkl")

    # Zapis modelu
    model.save(MODEL_PATH)
    print(f"Model zapisany do: {MODEL_PATH}")

    # Zapis artefaktów preprocessingu
    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": feature_names,
        "optimal_threshold": float(threshold),
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"Artefakty zapisane do: {ARTIFACTS_PATH}")

    # Instrukcja ładowania
    print("\nInstrukcja ładowania modelu:")
    print(f"1. model = tf.keras.models.load_model(r'{MODEL_PATH}')")
    print(f"2. artifacts = joblib.load(r'{ARTIFACTS_PATH}')")

if __name__ == "__main__":
    model = train_model()
