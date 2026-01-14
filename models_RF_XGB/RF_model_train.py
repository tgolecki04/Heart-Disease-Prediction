import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

from RF_model_preproces import (
    load_data,
    basic_eda,
    prepare_xy,
    train_test_split_stratified,
    simple_impute
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'RF_model', 'framingham_heart_study.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def main():

    # 1. Wczytanie danych
    df = load_data(DATA_PATH)
    df = df.drop(columns=['education'])
    basic_eda(df)

    # 2. Podział na X i y
    X, y = prepare_xy(df)

    # 3. Train-test split ze stratifikacją
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)

    # 4. Imputacja braków danych
    X_train_imp, X_test_imp, imputer = simple_impute(X_train, X_test)

    # 5. SMOTE — tylko na zbiorze treningowym
    print("\nBefore SMOTE:", np.bincount(y_train))
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_imp, y_train)
    print("After SMOTE:", np.bincount(y_train_res))

    # 6. Random Forest — baseline
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res)

    # Predykcja
    prob_rf = rf.predict_proba(X_test_imp)[:, 1]
    pred_rf = rf.predict(X_test_imp)

    # Raport
    print("\n--- RandomForest ---")
    print("AUC:", roc_auc_score(y_test, prob_rf))
    print(classification_report(y_test, pred_rf))

    # 7. Zapis modelu i imputera
    joblib.dump(
        {'model': rf, 'imputer': imputer},
        os.path.join(MODEL_DIR, 'rf_baseline.pkl')
    )

    print("\nModel zapisany w:", MODEL_DIR)


if __name__ == "__main__":
    main()
