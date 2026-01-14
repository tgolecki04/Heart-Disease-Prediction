import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# importujemy funkcje z preprocess.py
from RF_model_preproces import load_data, basic_eda, prepare_xy, train_test_split_stratified, simple_impute

DATA_REL_PATH = os.path.join(os.path.dirname(__file__), '..', 'RF_model', 'framingham_heart_study.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

USE_SMOTE = True
SAVE_SHAP = True
RANDOM_STATE = 123

# hiperparametry baseline XGBoost
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight = 5.58
)

def run():
    print("==> 1) Load data")
    df = load_data(DATA_REL_PATH)
    df = df.drop(columns=['education'])
    basic_eda(df)

    # 2) przygotowanie X, y
    print("\n==> 2) Prepare X, y")
    X, y = prepare_xy(df)
    print("Features shape:", X.shape)

    # 3) split train/test (stratified)
    print("\n==> 3) Train/test split (stratified)")
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # 4) imputacja (mediana) - uczymy imputera tylko na train
    print("\n==> 4) Imputation (median)")
    X_train_imp, X_test_imp, imputer = simple_impute(X_train, X_test)
    print("Missing values after imputation (train):\n", pd.DataFrame(X_train_imp).isna().sum().sum())
    print("Missing values after imputation (test):\n", pd.DataFrame(X_test_imp).isna().sum().sum())

    # 5) balansowanie klas (opcjonalnie SMOTE)
    if USE_SMOTE:
        print("\n==> 5) Balansowanie: SMOTE on TRAIN only")
        print("Before SMOTE:", np.bincount(y_train))
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train_imp, y_train)
        print("After SMOTE:", np.bincount(y_train_res))
    else:
        print("\n==> 5) No SMOTE: using original train distribution")
        X_train_res, y_train_res = X_train_imp, y_train

    # 6) Trening XGBoost (baseline)
    print("\n==> 6) Training XGBoost (baseline)")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train_res, y_train_res)

    # 7) Predykcje i ocena
    print("\n==> 7) Prediction & Evaluation")
    prob = xgb.predict_proba(X_test_imp)[:, 1]
    pred = xgb.predict(X_test_imp)
    auc = roc_auc_score(y_test, prob)
    print(f"AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))

    # 8) Zapis modelu + imputera
    print("\n==> 8) Save model and imputer")
    model_artifact = {
        'model': xgb,
        'imputer': imputer,
        'use_smote': USE_SMOTE,
        'features': list(X.columns)
    }
    model_path = os.path.join(MODEL_DIR, 'xgb_baseline_joblib.pkl')
    joblib.dump(model_artifact, model_path)
    print("Saved model to:", model_path)

    # 9) (opcjonalnie) SHAP
    if SAVE_SHAP:
        try:
            import shap
            import matplotlib.pyplot as plt

            print("\n==> 9) Generating SHAP summary plot (may take a while)")
            # przygotuj próbkę (np. do 2000 rekordów) żeby SHAP nie zjadał pamięci
            sample = X_test_imp if X_test_imp.shape[0] <= 2000 else X_test_imp.sample(2000, random_state=RANDOM_STATE)

            # TreeExplainer dla XGBoost
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(sample)

            # wykres i zapis
            plt.figure(figsize=(8,6))
            shap.summary_plot(shap_values, sample, feature_names=list(sample.columns), show=False)
            shap_plot_path = os.path.join(MODEL_DIR, 'shap_summary.png')
            plt.savefig(shap_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            print("SHAP summary plot saved to:", shap_plot_path)
        except Exception as e:
            print("Could not create SHAP plot (is shap installed and working?). Error:", str(e))

    print("\n==> Done")

if __name__ == "__main__":
    run()
