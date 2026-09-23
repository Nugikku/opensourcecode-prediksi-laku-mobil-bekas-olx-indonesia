import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

TANGGAL_HARI_INI = datetime.now().strftime("%d-%m-%Y")
OUTPUT_DIR = f"output_klasifikasi_{TANGGAL_HARI_INI}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

print("Membaca data bersih...")
file_dataset = "dataset_olx_bersih.csv"
if not os.path.exists(file_dataset): 
    raise FileNotFoundError(f"File '{file_dataset}' tidak ditemukan. Jalankan processing.py terlebih dahulu.")

df = pd.read_csv(file_dataset)

categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
numerical_cols = ['harga', 'tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun', 'deviasi_harga_pasar']
fitur = categorical_cols + numerical_cols
target = 'kategori_penjualan'

df_model = df.dropna(subset=fitur + [target]).copy()
y = df_model[target]
X = df_model[fitur]

daftar_kelas_unik = np.unique(y)
print(f"Data latih: {len(df_model)} baris")
print(f"Distribusi Target: {y.value_counts().to_dict()}")

# Gunakan stratify jika setiap kelas memiliki minimal 2 sampel
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=min(5, len(X_train))),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

hasil_metrik = []
best_model_name, best_model_pipeline, best_f1 = None, None, -1.0

print("\n" + "=" * 65)
print("HASIL KOMPARASI PERFORMA MODEL (3 KELAS)")
print("=" * 65)

for nama_model, classifier in models.items():
    pipeline = Pipeline(steps=[('prep', preprocessor), ('clf', classifier)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Penanganan aman penghitungan ROC-AUC multi-class
    auc = 0.5
    if hasattr(pipeline.named_steps['clf'], "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
            auc = roc_auc_score(
                y_test, 
                y_prob, 
                labels=pipeline.named_steps['clf'].classes_, 
                multi_class="ovr", 
                average="weighted"
            )
        except Exception:
            auc = 0.5

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nModel: {nama_model}")
    print(f"  Akurasi   : {acc * 100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    hasil_metrik.append({
        "Model": nama_model,
        "Akurasi": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    if f1 > best_f1:
        best_f1, best_model_name, best_model_pipeline = f1, nama_model, pipeline

print("\n" + "=" * 65)
print(f"Model Terbaik Terpilih: {best_model_name} (F1-Score: {best_f1:.4f})")
print("=" * 65)

df_metrik = pd.DataFrame(hasil_metrik)

with open("model_klasifikasi_penjualan.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)
print(f"[SUKSES] Model terbaik disimpan ke 'model_klasifikasi_penjualan.pkl'")

# Visualisasi Evaluasi
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_metrik))
lebar = 0.35
ax.bar(x - lebar/2, df_metrik["Akurasi"], lebar, label="Akurasi", color="#3B82F6")
ax.bar(x + lebar/2, df_metrik["F1-Score"], lebar, label="F1-Score (Weighted)", color="#10B981")
ax.set_xticks(x)
ax.set_xticklabels(df_metrik["Model"], rotation=15, ha="right")
ax.set_ylim(0, 1.15)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "komparasi_metrik.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_model_pipeline, X_test, y_test, cmap="Blues", ax=ax)
ax.set_title(f"Confusion Matrix (3 Kelas) — {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close(fig)

df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik.csv"), index=False)
print(f"Grafik dan tabel evaluasi tersimpan di folder '{OUTPUT_DIR}/'.")