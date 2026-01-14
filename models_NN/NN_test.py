import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("framingham_heart_study.csv")
df = df.drop(columns=['education'])

# Uzupełnienie medianą NA
df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Split danych
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Skalowanie danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ==============================
# TESTOWANIE ARCHITEKTUR MLP
# ==============================

best_score = 0
best_params = None
best_recall = 0
best_precision = 0

layer_configs = [
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
    (1024, 512),

    (16, 8, 4),
    (32, 16, 8),
    (64, 32, 16),
    (128, 64, 32),
    (256, 128, 64),
    (512, 256, 128),
    (1024, 512, 256),

    (32, 16, 8, 4),
    (64, 32, 16, 8),
    (128, 64, 32, 16),
    (256, 128, 64, 32),
    (512, 256, 128, 64),
    (1024, 512, 256, 128)
]

print("\n==============================")
print(" ROZPOCZYNAM TEST ARCHITEKTUR")
print("==============================\n")

for layers in layer_configs:

    print(f"Test: {layers}")

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=320,
        random_state=42
    )

    # Wagi dla klas
    weights = np.where(y_train_sm == 1, 20, 1)

    model.fit(X_train_sm, y_train_sm, sample_weight=weights)

    # Prawdopodobieństwa
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.2
    y_pred = (y_proba >= threshold).astype(int)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    if precision != 0:
        ratio = precision / recall
    else:
        ratio = 0

    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall / Precision: {ratio:.4f}\n")

    if ratio > best_score:
        best_score = ratio
        best_params = layers
        best_recall = recall
        best_precision = precision

print("====================================")
print(" NAJLEPSZA ARCHITEKTURA MLP")
print("====================================")
print(f"Warstwy: {best_params}")
print(f"Recall: {best_recall}")
print(f"Precision: {best_precision}")
print(f"Recall / Precision: {best_score}")

