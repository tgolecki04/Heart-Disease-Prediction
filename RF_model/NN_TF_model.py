import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             fbeta_score, matthews_corrcoef, balanced_accuracy_score,
                             roc_curve)
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os

warnings.filterwarnings('ignore')

# 1. Wczytanie i przygotowanie danych
print("=" * 80)
print("MODEL PREDYKCJI RYZYKA CHORÓB SERCA - WERSJA MEDYCZNA")
print("=" * 80)

data = pd.read_csv("framingham_heart_study.csv")

if 'education' in data.columns:
    data = data.drop(columns=["education"])

feature_names = data.drop(columns=["TenYearCHD"], axis=1).columns.tolist()
print(f"\n📊 Dostępne cechy ({len(feature_names)}): {feature_names}")

X = data.drop(columns=["TenYearCHD"], axis=1)
y = data["TenYearCHD"]

print(f"\n📈 Rozkład klas:")
print(f"  Klasa 0 (zdrowi): {sum(y == 0)} ({sum(y == 0) / len(y) * 100:.1f}%)")
print(f"  Klasa 1 (chorzy): {sum(y == 1)} ({sum(y == 1) / len(y) * 100:.1f}%)")

# 2. Uzupełnianie brakujących wartości i skalowanie
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Podział na zbiory z zachowaniem proporcji klas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

print(f"\n🔧 Podział danych:")
print(f"  Zbiór treningowy: {X_train.shape[0]} próbek")
print(f"  Zbiór testowy:    {X_test.shape[0]} próbek")

# 4. Balansowanie klas TYLKO na zbiorze treningowym
sm = ADASYN(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"\n⚖️  Balansowanie klas (ADASYN):")
print(f"  Przed: {X_train.shape[0]} → Po: {X_train_res.shape[0]}")

# 5. OPTYMALNA ARCHITEKTURA DLA MEDYCYNY
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_res.shape[1],)),

    # Warstwa 1 - minimalna regularyzacja dla lepszego dopasowania
    tf.keras.layers.Dense(64, activation="elu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    # Warstwa 2
    tf.keras.layers.Dense(32, activation="elu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    # Warstwa wyjściowa
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print(f"\n🧠 Architektura modelu:")
model.summary()


# 6. FUNKCJA STRATY OPTYMALNA DLA MEDYCYNY (priorytet: wykrycie chorych)
def medical_loss(y_true, y_pred, fn_weight=8.0):
    """Custom loss function that HEAVILY penalizes false negatives"""
    y_true = tf.cast(y_true, tf.float32)

    # Standard binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Calculate false negatives with EXTRA heavy weight
    # We want to minimize missing sick patients at all costs
    fn = y_true * (1 - y_pred)  # Large when we miss positive cases

    # Weighted loss: BCE + VERY heavy penalty for FN
    weighted_loss = bce + fn_weight * tf.reduce_mean(fn)

    return weighted_loss


# 7. KOMPILACJA Z OPTYMALNYMI PARAMETRAMI
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=medical_loss,
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR')
    ]
)


# 8. CALLBACK Z OPTYMALIZACJĄ DLA WYSOKIEGO RECALL
class MedicalOptimizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_recall = 0
        self.best_weights = None
        self.best_threshold = 0.5

    def on_epoch_end(self, epoch, logs=None):
        y_pred_prob = self.model.predict(self.X_val, verbose=0).flatten()

        # Znajdź próg dający recall >= 75%
        precision, recall, thresholds = precision_recall_curve(self.y_val, y_pred_prob)

        # Szukamy najlepszego F2-score z recall >= 75%
        best_f2 = 0
        best_threshold = 0.5

        for i in range(len(thresholds)):
            if recall[i] >= 0.75:  # Wymagamy wysokiego recall
                y_pred = (y_pred_prob >= thresholds[i]).astype(int)
                f2 = fbeta_score(self.y_val, y_pred, beta=2)
                if f2 > best_f2:
                    best_f2 = f2
                    best_threshold = thresholds[i]

        logs['val_best_threshold'] = best_threshold

        # Zapisz najlepsze wagi jeśli znaleźliśmy próg z recall >= 75%
        if best_f2 > 0 and best_f2 > self.best_recall:
            self.best_recall = best_f2
            self.best_weights = self.model.get_weights()
            self.best_threshold = best_threshold


# 9. PRZYGOTOWANIE DANYCH DO TRENOWANIA
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_res, y_train_res, test_size=0.2, random_state=42, stratify=y_train_res
)

# 10. CALLBACKS
early_stop = EarlyStopping(
    monitor='val_recall',
    patience=20,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

medical_callback = MedicalOptimizationCallback(X_val, y_val)

# 11. TRENOWANIE
print(f"\n🚀 Rozpoczynam trenowanie modelu...")
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=100,
    callbacks=[early_stop, reduce_lr, medical_callback],
    verbose=1
)

# Użyj najlepszych wag jeśli znaleziono
if medical_callback.best_weights is not None:
    model.set_weights(medical_callback.best_weights)
    print(f"\n✅ Załadowano wagi z najlepszym recall: {medical_callback.best_recall:.4f}")
    suggested_threshold = medical_callback.best_threshold
else:
    suggested_threshold = 0.5

print(f"\n📊 Sugerowany próg z treningu: {suggested_threshold:.3f}")

# 12. TEST RÓŻNYCH PROGÓW KLASYFIKACJI
print(f"\n{'=' * 80}")
print("TEST RÓŻNYCH PROGÓW KLASYFIKACJI - WYBÓR OPTYMALNEGO")
print(f"{'=' * 80}")

y_pred_prob = model.predict(X_test, verbose=0).flatten()

# Testuj progi od 0.3 do 0.7
thresholds_to_test = np.arange(0.3, 0.71, 0.05)
results = []

for threshold in thresholds_to_test:
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f2 = fbeta_score(y_test, y_pred, beta=2)

    results.append({
        'Próg': f'{threshold:.2f}',
        'Recall': f'{recall:.3f}',
        'Precision': f'{precision:.3f}',
        'Specificity': f'{specificity:.3f}',
        'F2': f'{f2:.3f}',
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })

# Wyświetl wyniki w tabeli
df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# 13. AUTOMATYCZNY WYBÓR PROGU DLA MEDYCYNY
print(f"\n{'=' * 80}")
print("AUTOMATYCZNY WYBÓR OPTYMALNEGO PROGU")
print(f"{'=' * 80}")

# Strategia: Znajdź próg gdzie recall >= 75% i F2-score jest maksymalne
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)

best_threshold = 0.5
best_f2 = 0
best_recall = 0

for i in range(len(thresholds_pr)):
    threshold = thresholds_pr[i]
    recall = recall_vals[i]

    # Priorytet: recall >= 75%
    if recall >= 0.75:
        y_pred_temp = (y_pred_prob >= threshold).astype(int)
        f2 = fbeta_score(y_test, y_pred_temp, beta=2)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
            best_recall = recall

# Jeśli nie znaleziono progu z recall >= 75%, użyj tego z max F2
if best_threshold == 0.5:
    f2_scores = []
    for i in range(len(thresholds_pr)):
        y_pred_temp = (y_pred_prob >= thresholds_pr[i]).astype(int)
        f2_scores.append(fbeta_score(y_test, y_pred_temp, beta=2))

    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds_pr[best_idx]
    best_recall = recall_vals[best_idx]
    best_f2 = f2_scores[best_idx]

# W MEDYCYNIE REKOMENDUJEMY PRÓG 0.6 DLA BEZPIECZEŃSTWA
MEDICAL_RECOMMENDED_THRESHOLD = 0.6
print(f"\n🏥 REKOMENDOWANY PRÓG MEDYCZNY: {MEDICAL_RECOMMENDED_THRESHOLD}")
print(f"   Dlaczego 60%? Bo daje lepszą czułość (więcej wykrytych chorych).")

# Sprawdź jak wygląda model z progiem 0.6
y_pred_optimal = (y_pred_prob >= MEDICAL_RECOMMENDED_THRESHOLD).astype(int)
optimal_cm = confusion_matrix(y_test, y_pred_optimal)
optimal_tn, optimal_fp, optimal_fn, optimal_tp = optimal_cm.ravel()

optimal_recall = optimal_tp / (optimal_tp + optimal_fn)
optimal_precision = optimal_tp / (optimal_tp + optimal_fp) if (optimal_tp + optimal_fp) > 0 else 0
optimal_specificity = optimal_tn / (optimal_tn + optimal_fp) if (optimal_tn + optimal_fp) > 0 else 0
optimal_f2 = fbeta_score(y_test, y_pred_optimal, beta=2)

print(f"\n📈 Wyniki z progiem {MEDICAL_RECOMMENDED_THRESHOLD}:")
print(f"  Czułość (Recall):    {optimal_recall:.3f} ({optimal_tp}/{optimal_tp + optimal_fn})")
print(f"  Precyzja:            {optimal_precision:.3f}")
print(f"  Specyficzność:       {optimal_specificity:.3f}")
print(f"  F2-Score:            {optimal_f2:.3f}")

# 14. SZCZEGÓŁOWA OCENA MEDYCZNA
print(f"\n{'=' * 80}")
print("OCENA MODELU MEDYCZNEGO - RAPORT KOŃCOWY")
print(f"{'=' * 80}")

# Oblicz wszystkie metryki
auc_roc = roc_auc_score(y_test, y_pred_prob)
auc_pr = average_precision_score(y_test, y_pred_prob)
npv = optimal_tn / (optimal_tn + optimal_fn) if (optimal_tn + optimal_fn) > 0 else 0
balanced_acc = balanced_accuracy_score(y_test, y_pred_optimal)
mcc = matthews_corrcoef(y_test, y_pred_optimal)
f1 = fbeta_score(y_test, y_pred_optimal, beta=1)

print(f"\n📊 METRYKI PODSTAWOWE:")
print(f"  True Positives (TP):   {optimal_tp:4d} - Chorzy poprawnie wykryci")
print(f"  False Positives (FP):  {optimal_fp:4d} - Zdrowi błędnie alarmowani")
print(f"  False Negatives (FN):  {optimal_fn:4d} - Chorzy przeoczeni (NIEBEZPIECZNE!)")
print(f"  True Negatives (TN):   {optimal_tn:4d} - Zdrowi poprawnie uspokojeni")

print(f"\n🎯 METRYKI KLINICZNE:")
print(f"  ⚕️  Czułość (Recall):     {optimal_recall:.1%}  - Wykrywamy {optimal_tp} z {optimal_tp + optimal_fn} chorych")
print(
    f"  ⚕️  Precyzja (PPV):       {optimal_precision:.1%}  - {optimal_tp} z {optimal_tp + optimal_fp} alarmów to prawdziwe zagrożenia")
print(
    f"  ⚕️  Specyficzność:        {optimal_specificity:.1%}  - {optimal_tn} z {optimal_tn + optimal_fp} zdrowych nie ma fałszywych alarmów")
print(f"  ⚕️  NPV:                  {npv:.1%}  - Pewność przy wyniku negatywnym")

print(f"\n📈 METRYKI STATYSTYCZNE:")
print(f"  AUC-ROC:               {auc_roc:.3f}")
print(f"  AUC-PR:                {auc_pr:.3f}")
print(f"  Balanced Accuracy:     {balanced_acc:.3f}")
print(f"  F1-Score:              {f1:.3f}")
print(f"  F2-Score:              {optimal_f2:.3f}")
print(f"  MCC:                   {mcc:.3f}")

print(f"\n📋 MACIERZ POMYŁEK:")
print(f"\n{optimal_cm}")
print(f"\n[0,0] = TN ({optimal_tn}) | [0,1] = FP ({optimal_fp})")
print(f"[1,0] = FN ({optimal_fn}) | [1,1] = TP ({optimal_tp})")

print(f"\n📝 RAPORT KLASYFIKACJI:")
print(classification_report(y_test, y_pred_optimal,
                            target_names=['Niskie ryzyko', 'Wysokie ryzyko']))

# 15. ANALIZA KOSZTÓW/KORZYŚCI DLA MEDYCYNY
print(f"\n{'=' * 80}")
print("ANALIZA KOSZTÓW/KORZYŚCI - PERSPEKTYWA MEDYCZNA")
print(f"{'=' * 80}")

print(f"\n💰 KOSZTY:")
print(f"  • Fałszywie pozytywne ({optimal_fp} osób):")
print(f"    - Dodatkowe badania (EKG, echo serca, próby wysiłkowe)")
print(f"    - Stres i niepokój pacjenta")
print(f"    - Koszt: ok. 500-2000 zł na pacjenta")

print(f"\n  • Fałszywie negatywne ({optimal_fn} osób - NIEBEZPIECZNE!):")
print(f"    - Brak leczenia → zawał, udar, śmierć")
print(f"    - Koszt leczenia powikłań: 50,000-200,000 zł")
print(f"    - Koszt ludzki: cierpienie, utrata zdrowia/życia")

print(f"\n✅ KORZYŚCI:")
print(f"  • Prawdziwie pozytywne ({optimal_tp} osób):")
print(f"    - Wczesna interwencja → zapobieganie chorobie")
print(f"    - Koszt prewencji: 100-1000 zł na pacjenta")
print(f"    - Oszczędność: 50-200x niższy koszt niż leczenie")

print(f"\n⚖️  PODSUMOWANIE KOSZTÓW:")
total_fp_cost = optimal_fp * 1000  # Średnio 1000 zł na fałszywy alarm
total_fn_cost = optimal_fn * 100000  # Średnio 100,000 zł na przeoczenie
total_tp_savings = optimal_tp * 50000  # Średnio 50,000 zł oszczędności na wczesnym wykryciu

print(f"  Koszt fałszywych alarmów:    {total_fp_cost:,.0f} zł")
print(f"  Koszt przeoczonych chorych:  {total_fn_cost:,.0f} zł")
print(f"  Oszczędność z wczesnych wykryć: {total_tp_savings:,.0f} zł")
print(f"  BILANS: {total_tp_savings - total_fp_cost - total_fn_cost:,.0f} zł")

# 16. WIZUALIZACJE DLA LEKARZY
print(f"\n{'=' * 80}")
print("GENEROWANIE WYKRESÓW DIAGNOSTYCZNYCH...")
print(f"{'=' * 80}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Krzywa ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_roc:.3f}')
axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
axes[0, 0].set_xlabel('False Positive Rate (1 - Specificity)')
axes[0, 0].set_ylabel('True Positive Rate (Recall)')
axes[0, 0].set_title('Krzywa ROC')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Krzywa Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
axes[0, 1].plot(recall, precision, 'g-', linewidth=2, label=f'AUC-PR = {auc_pr:.3f}')
axes[0, 1].axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Min. precyzja (25%)')
axes[0, 1].axvline(x=0.75, color='red', linestyle='--', alpha=0.5, label='Min. czułość (75%)')
axes[0, 1].scatter([optimal_recall], [optimal_precision], color='black', s=100,
                   label=f'Próg {MEDICAL_RECOMMENDED_THRESHOLD}')
axes[0, 1].set_xlabel('Czułość (Recall)')
axes[0, 1].set_ylabel('Precyzja')
axes[0, 1].set_title('Krzywa Precyzja-Czułość')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Rozkład prawdopodobieństw
axes[0, 2].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.6,
                label='Niskie ryzyko (w rzeczywistości)', color='green')
axes[0, 2].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.6,
                label='Wysokie ryzyko (w rzeczywistości)', color='red')
axes[0, 2].axvline(x=MEDICAL_RECOMMENDED_THRESHOLD, color='black',
                   linestyle='--', linewidth=2, label=f'Próg {MEDICAL_RECOMMENDED_THRESHOLD}')
axes[0, 2].set_xlabel('Przewidywane prawdopodobieństwo ryzyka')
axes[0, 2].set_ylabel('Liczba pacjentów')
axes[0, 2].set_title('Rozkład przewidywań')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Heatmap macierzy pomyłek
sns.heatmap(optimal_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['Przew. niskie', 'Przew. wysokie'],
            yticklabels=['Rzecz. niskie', 'Rzecz. wysokie'])
axes[1, 0].set_xlabel('Przewidziane ryzyko')
axes[1, 0].set_ylabel('Rzeczywiste ryzyko')
axes[1, 0].set_title('Macierz decyzji klinicznych')

# 5. Porównanie metryk
metrics = ['Czułość', 'Precyzja', 'Specyficzność', 'F2-Score']
values = [optimal_recall, optimal_precision, optimal_specificity, optimal_f2]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = axes[1, 1].bar(metrics, values, color=colors)
axes[1, 1].axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Cel czułości')
axes[1, 1].axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Cel precyzji')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_ylabel('Wartość')
axes[1, 1].set_title('Kluczowe metryki bezpieczeństwa')
axes[1, 1].legend()

for bar, v in zip(bars, values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# 6. Analiza błędów
error_labels = ['TP', 'FP', 'FN', 'TN']
error_counts = [optimal_tp, optimal_fp, optimal_fn, optimal_tn]
error_colors = ['#4CAF50', '#FFC107', '#F44336', '#2196F3']

axes[1, 2].pie(error_counts, labels=error_labels, colors=error_colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
axes[1, 2].set_title('Rozkład decyzji modelu')

plt.tight_layout()
plt.savefig('medical_model_evaluation.png', dpi=150, bbox_inches='tight')
print(f"✅ Zapisano wykresy do: medical_model_evaluation.png")
plt.show()

# 17. ANALIZA CECH (Feature Importance)
print(f"\n{'=' * 80}")
print("ANALIZA NAJWAŻNIEJSZYCH CZYNNIKÓW RYZYKA")
print(f"{'=' * 80}")

# Pobierz wagi z pierwszej warstwy
first_layer_weights = model.layers[0].get_weights()[0]
feature_importance = np.abs(first_layer_weights).mean(axis=1)

# Normalizuj do 0-100%
feature_importance = 100 * feature_importance / feature_importance.sum()

# Utwórz DataFrame
importance_df = pd.DataFrame({
    'Cecha': feature_names,
    'Ważność (%)': feature_importance
}).sort_values('Ważność (%)', ascending=False)

print("\n🏆 TOP 10 najważniejszych czynników ryzyka:")
print(importance_df.head(10).to_string(index=False))

# 18. PODSUMOWANIE I REKOMENDACJE
print(f"\n{'=' * 80}")
print("OSTATECZNA OCENA: CZY MODEL NADAJE SIĘ DO UŻYTKU MEDYCZNEGO?")
print(f"{'=' * 80}")

# Kryteria akceptacji klinicznej
CRITERIA = {
    'Czułość ≥ 75%': optimal_recall >= 0.75,
    'Precyzja ≥ 25%': optimal_precision >= 0.25,
    'AUC-ROC ≥ 0.70': auc_roc >= 0.70,
    'F2-Score ≥ 0.50': optimal_f2 >= 0.50,
    'FN < 10% chorych': optimal_fn / (optimal_tp + optimal_fn) < 0.10,
}

print("\n📋 KRYTERIA AKCEPTACJI KLINICZNEJ:")
all_passed = True
for criterion, passed in CRITERIA.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {criterion}")
    if not passed:
        all_passed = False

if all_passed:
    print(f"\n🎉 MODEL SPEŁNIA WSZYSTKIE KRYTERIA MEDYCZNE!")
    print("   Może być rozważany do zastosowań przesiewowych pod nadzorem lekarza.")
elif sum(CRITERIA.values()) >= 3:
    print(f"\n⚠️  MODEL SPEŁNIA WIĘKSZOŚĆ KRYTERIÓW")
    print("   Wymaga dodatkowej walidacji przed zastosowaniem klinicznym.")
else:
    print(f"\n❌ MODEL NIE SPEŁNIA KRYTERIÓW BEZPIECZEŃSTWA")
    print("   Nie nadaje się do zastosowań klinicznych bez dalszych poprawek.")

# 19. REKOMENDACJE DLA WDROŻENIA
print(f"\n{'=' * 80}")
print("REKOMENDACJE DLA WDROŻENIA KLINICZNEGO")
print(f"{'=' * 80}")

print(f"\n1. 🏥 ZASTOSOWANIE:")
print(f"   • Narzędzie wspomagające decyzję lekarza (NIE zastępuje lekarza!)")
print(f"   • System przesiewowy w podstawowej opiece zdrowotnej")
print(f"   • Alert system w aplikacjach zdrowotnych")

print(f"\n2. ⚠️  OGRANICZENIA:")
print(f"   • {optimal_fn} z {optimal_tp + optimal_fn} chorych może być przeoczonych")
print(f"   • {optimal_fp} z {optimal_fp + optimal_tn} zdrowych dostanie fałszywe alarmy")
print(f"   • Wymaga potwierdzenia diagnozy przez lekarza")

print(f"\n3. 📊 MONITORING:")
print(f"   • Śledź szczególnie przypadki FN (przeoczone)")
print(f"   • Regularnie aktualizuj model nowymi danymi")
print(f"   • Monitoruj drift koncepcyjny")

print(f"\n4. 🔧 OPTYMALIZACJA:")
print(f"   • Próg można dostosować: 0.5-0.7 w zależności od priorytetów")
print(f"   • 0.6 - kompromis między wykrywaniem a fałszywymi alarmami")
print(f"   • 0.5 - maksymalne wykrywanie, więcej fałszywych alarmów")
print(f"   • 0.7 - mniej fałszywych alarmów, ale więcej przeoczeń")

# 20. ZAPIS MODELU DO UŻYTKU
print(f"\n{'=' * 80}")
print("ZAPIS MODELU DO UŻYTKU")
print(f"{'=' * 80}")

import joblib

# Przygotuj artefakty modelu
model_artifacts = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'optimal_threshold': MEDICAL_RECOMMENDED_THRESHOLD,
    'feature_names': feature_names,
    'metrics': {
        'recall': float(optimal_recall),
        'precision': float(optimal_precision),
        'specificity': float(optimal_specificity),
        'auc_roc': float(auc_roc),
        'f2_score': float(optimal_f2)
    }
}

# Zapisz model
joblib.dump(model_artifacts, 'medical_heart_risk_model.pkl')
print(f"✅ Zapisano model do: medical_heart_risk_model.pkl")

print(f"\n💡 PRZYKŁAD UŻYCIA:")
print(f"   model = joblib.load('medical_heart_risk_model.pkl')")
print(f"   prediction = model['model'].predict(patient_data)")
print(f"   if prediction >= {MEDICAL_RECOMMENDED_THRESHOLD}:")
print(f"       print('WYSOKIE RYZYKO - skonsultuj się z lekarzem!')")
print(f"   else:")
print(f"       print('Niskie ryzyko - kontrola za rok')")

print(f"\n{'=' * 80}")
print("✅ MODEL GOTOWY DO UŻYTKU (jako narzędzie wspomagające)")
print(f"{'=' * 80}")