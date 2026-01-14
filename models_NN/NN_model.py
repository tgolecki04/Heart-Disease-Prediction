import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("framingham_heart_study.csv")
df = df.drop(columns=['education'])

#Uzupelniene mediana NA
df = df.fillna(df.median(numeric_only=True))
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Split danych
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

#Skalowanie danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#SMOTHE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

#Sieć neuronowa MLP (MultiLayer Perceptron)
#Próa dla 3 warstw oraz rozkladu warstw 128/64/32
model = MLPClassifier(
    hidden_layer_sizes=(32,16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=320,
    random_state=42
)
weights = np.where(y_train_sm==1, 10, 1)
model.fit(X_train_sm, y_train_sm)

# Wyliczanie prawdopodobieństw
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.2 # im niżej, tym wyższy recall!!!
y_pred = (y_proba >= threshold).astype(int)

#Wyniki
recall = recall_score(y_test, y_pred)
print(f"\nRecall (threshold={threshold}):", recall)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))






















