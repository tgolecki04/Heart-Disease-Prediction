import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv("framingham_heart_study.csv")
data = data.drop(columns=["education"])

X = data.drop(columns=["TenYearCHD"], axis=1)
y = data["TenYearCHD"]

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_res.shape[1],)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

train = model.fit(
    X_train_res, y_train_res,
    batch_size=16,
    epochs=60,
    validation_split=0.2,
    callbacks=[early_stop]
)

y_pred_prob = model.predict(X_test)
threshold = 0.3
y_pred = (y_pred_prob > threshold).astype(int)

loss, acc, recall, precision = model.evaluate(X_test, y_test)
print(f"\nStrata (loss): {loss:.4f}")
print(f"Dokładność (accuracy): {acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {auc:.3f}")

sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)
print(f"\nPrzykładowa predykcja: {float(pred[0][0])*100:.2f}%")

plt.figure(figsize=(10,4))
plt.plot(train.history['loss'], label='train loss')
plt.plot(train.history['val_loss'], label='val loss')
plt.plot(train.history['accuracy'], label='train acc')
plt.plot(train.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Historia uczenia")
plt.show()
