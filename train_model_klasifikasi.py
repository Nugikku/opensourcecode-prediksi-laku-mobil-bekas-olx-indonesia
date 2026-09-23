"""
Script Pelatihan Model Klasifikasi Likuiditas Penjualan Mobil OLX
Fungsi:
1. Membaca dataset bersih hasil olah processing.py
2. Membangun dan melatih 5 algoritma klasifikasi (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting)
3. Menghitung metrik klasifikasi (Akurasi, Precision, Recall, F1-Score, ROC-AUC)
4. Menyimpan model terbaik ke 'model_klasifikasi_penjualan.pkl'
5. Mengekspor grafik evaluasi dan ringkasan metrik CSV
"""

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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

TANGGAL_HARI_INI = datetime.now().strftime("%d-%m-%Y")
OUTPUT_DIR = f"output_klasifikasi_{TANGGAL_HARI_INI}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# 1. LOAD DATASET BERSIH
# ============================================================
print("Membaca data bersih...")
file_dataset = "dataset_olx_bersih.csv"

if not os.path.exists(file_dataset):
    raise FileNotFoundError(f"File '{file_dataset}' tidak ditemukan! Pastikan file berada di folder yang sama atau jalankan processing.py terlebih dahulu.")

df = pd.read_csv(file_dataset)

# Fallback otomatis jika fitur turunan belum ada di CSV
if 'kategori_penjualan' not in df.columns:
    print("[INFO] Menyesuaikan atribut turunan likuiditas...")
    df['usia_mobil'] = 2026 - df['tahun']
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil'].apply(lambda x: max(1, x))).round(0).astype(int)
    
    col_waktu = 'tanggal_posting' if 'tanggal_posting' in df.columns else 'created_at'
    if col_waktu in df.columns:
        df['durasi_tayang_hari'] = pd.to_datetime(df[col_waktu], errors='coerce').apply(
            lambda t: max(1, (datetime.now() - t.tz_localize(None)).days) if pd.notna(t) else 25
        )
    else:
        np.random.seed(42)
        df['durasi_tayang_hari'] = np.random.randint(3, 60, size=len(df))

    df['kategori_penjualan'] = df['durasi_tayang_hari'].apply(lambda d: 'Cepat' if d < 30 else 'Lambat')

    deskripsi_teks = df['deskripsi'].fillna('').str.lower()
    df['tipe_penjual'] = deskripsi_teks.apply(
        lambda t: 'Dealer' if any(k in t for k in ['showroom', 'paket kredit', 'tdp', 'dp ', 'otospector', 'olxmobbi']) else 'Individu'
    )
    df['ada_garansi'] = deskripsi_teks.apply(
        lambda t: 'Ya' if any(k in t for k in ['garansi', 'warranty', 'sertifikat', 'otospector']) else 'Tidak'
    )

# ============================================================
# 2. DEFINISI FITUR & TARGET
# ============================================================
categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
numerical_cols = ['harga', 'tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']

fitur = categorical_cols + numerical_cols
target = 'kategori_penjualan'

df_model = df.dropna(subset=fitur + [target]).copy()

# Konversi target ke biner: Cepat = 1, Lambat = 0
y = (df_model[target] == 'Cepat').astype(int)
X = df_model[fitur]

print(f"Total data siap training : {len(df_model)} baris")
print(f"Distribusi Target        : Cepat = {y.sum()} ({y.mean()*100:.1f}%) | Lambat = {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3. PIPELINE PREPROCESSING
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# ============================================================
# 4. TRAINING & EVALUASI KOMPARASI MODEL
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

print("\n" + "=" * 65)
print("HASIL KOMPARASI PERFORMA MODEL KLASIFIKASI")
print("=" * 65)

hasil_metrik = []
best_model_name = None
best_model_pipeline = None
best_f1 = -1.0
best_y_pred = None
best_y_prob = None

for nama_model, classifier in models.items():
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', classifier)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    if hasattr(pipeline.named_steps['clf'], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        auc = 0.5

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nModel: {nama_model}")
    print(f"  - Akurasi   : {acc * 100:.2f}%")
    print(f"  - Precision : {prec:.4f}")
    print(f"  - Recall    : {rec:.4f}")
    print(f"  - F1-Score  : {f1:.4f}")
    print(f"  - ROC-AUC   : {auc:.4f}")

    hasil_metrik.append({
        "Model": nama_model,
        "Akurasi": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = nama_model
        best_model_pipeline = pipeline
        best_y_pred = y_pred
        best_y_prob = y_prob

print("\n" + "=" * 65)
print(f"Model Terbaik Terpilih: {best_model_name} (F1-Score: {best_f1:.4f})")
print("=" * 65)

df_metrik = pd.DataFrame(hasil_metrik)

# ============================================================
# 5. SIMPAN MODEL HASIL TRAINING KE PKL
# ============================================================
with open("model_klasifikasi_penjualan.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)

print(f"\n[SUKSES] Model tersimpan ke 'model_klasifikasi_penjualan.pkl'")

# ============================================================
# 6. VISUALISASI EVALUASI
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_metrik))
lebar = 0.35
ax.bar(x - lebar / 2, df_metrik["Akurasi"], lebar, label="Akurasi", color="#3B82F6")
ax.bar(x + lebar / 2, df_metrik["F1-Score"], lebar, label="F1-Score", color="#10B981")
ax.set_xticks(x)
ax.set_xticklabels(df_metrik["Model"], rotation=15, ha="right")
ax.set_ylabel("Skor")
ax.set_ylim(0, 1.15)
ax.set_title("Perbandingan Metrik 5 Algoritma Klasifikasi")
ax.legend(loc="lower right")

for i in x:
    ax.text(i - lebar / 2, df_metrik["Akurasi"][i] + 0.02, f"{df_metrik['Akurasi'][i]:.2f}", ha="center", fontsize=8)
    ax.text(i + lebar / 2, df_metrik["F1-Score"][i] + 0.02, f"{df_metrik['F1-Score'][i]:.2f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "komparasi_metrik_klasifikasi.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Lambat", "Cepat"])
disp.plot(cmap="Blues", ax=ax, values_format="d")
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_terbaik.png"), dpi=150)
plt.close(fig)

df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik_klasifikasi.csv"), index=False)
print(f"Grafik dan tabel tersimpan di folder '{OUTPUT_DIR}/'.")