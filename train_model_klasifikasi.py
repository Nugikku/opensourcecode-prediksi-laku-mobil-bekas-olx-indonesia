"""
AutoLiquid DSS: Pelatihan Model Klasifikasi Likuiditas (Cepat/Sedang/Lambat)
Arsitektur: Proxy Labeling (OOF Price Deviation, data-driven) + Classifier ML

Catatan Metodologi:
  Dataset OLX tidak menyediakan status "terjual" atau tanggal iklan turun
  (sudah difilter oleh platform), sehingga label likuiditas TIDAK BISA
  diperoleh langsung dari observasi historis. Sebagai gantinya, label
  dibentuk memakai pendekatan proxy/weak-labeling yang data-driven:

  1. Estimasi "harga wajar" tiap unit dihitung memakai prediksi
     out-of-fold (cross_val_predict, K-Fold) dari model regresi harga --
     prediksi harga wajar tiap baris TIDAK dilatih dari baris itu
     sendiri, sehingga tidak bocor (data leakage).
  2. Deviasi harga_aktual terhadap harga_wajar_oof dihitung utk seluruh
     dataset, lalu dibagi 3 kelas memakai TERTILE (persentil 33/67) dari
     distribusi deviasi tsb -- bukan angka ambang bebas -- supaya
     proporsi kelas seimbang & berdasar pada sebaran data itu sendiri.
  3. Sinyal tambahan (km/tahun tinggi & tidak ada garansi) menurunkan
     kelas "Cepat" jadi "Sedang" -- asumsi domain bahwa unit dgn
     pemakaian sangat tinggi tanpa garansi lebih sulit laku meski murah.
  4. Classifier ML kemudian DILATIH utk memprediksi label proxy ini,
     sehingga hasil akhirnya model klasifikasi yg dievaluasi dgn
     accuracy/F1/confusion matrix -- bukan aturan if-else manual.

  KETERBATASAN PENTING: label bersifat proxy, bukan status "terjual"
  historis sebenarnya. Selain itu, dataset sangat timpang antara mobil
  baru & lama (lihat diagnostik kecukupan sampel di bawah) -- kombinasi
  merek+model+usia yang jarang muncul di data TIDAK bisa diprediksi
  dengan andal walau modelnya sudah benar. Ambang & mekanisme peringatan
  keyakinan rendah untuk kasus ini ada di processing.py & diterapkan
  saat prediksi di test_prediksi_klasifikasi.py.
"""

import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

try:
    from processing import AMBANG_KEYAKINAN_SAMPEL
except ImportError:
    AMBANG_KEYAKINAN_SAMPEL = 15  # fallback jika processing.py tidak ditemukan di folder yang sama

TANGGAL_HARI_INI = datetime.now().strftime("%d-%m-%Y")
OUTPUT_DIR = f"output_klasifikasi_{TANGGAL_HARI_INI}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

print("Membaca dataset bersih...")
file_dataset = "dataset_olx_bersih.csv"
if not os.path.exists(file_dataset):
    raise FileNotFoundError(f"File '{file_dataset}' tidak ditemukan. Jalankan processing.py terlebih dahulu.")

df = pd.read_csv(file_dataset)

categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
numerical_cols = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']
fitur = categorical_cols + numerical_cols
target_harga = 'harga'

df_model = df.dropna(subset=fitur + [target_harga]).copy().reset_index(drop=True)
X = df_model[fitur]
y_harga = df_model[target_harga]

print(f"Total sampel : {len(df_model):,} baris")

# --- Diagnostik kecukupan sampel (untuk dilaporkan di BAB Hasil/Keterbatasan) ---
if 'jumlah_sampel_kategori' in df_model.columns:
    n_rendah = int((df_model['jumlah_sampel_kategori'] < AMBANG_KEYAKINAN_SAMPEL).sum())
    print(f"[DIAGNOSTIK] {n_rendah:,} dari {len(df_model):,} baris ({n_rendah/len(df_model)*100:.1f}%) berasal dari "
          f"kombinasi merek+model+usia dgn sampel < {AMBANG_KEYAKINAN_SAMPEL} -- performa model utk segmen ini "
          f"secara inheren kurang andal, terlepas dari algoritma yang dipakai.")

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# ============================================================
# TAHAP 1: ESTIMASI HARGA WAJAR OUT-OF-FOLD (basis pembuatan label proxy)
# ============================================================
print("\n" + "=" * 75)
print("TAHAP 1: ESTIMASI HARGA WAJAR (OUT-OF-FOLD, TANPA DATA LEAKAGE)")
print("=" * 75)

pipeline_harga_oof = Pipeline(steps=[
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
harga_wajar_oof = cross_val_predict(pipeline_harga_oof, X, y_harga, cv=kfold, n_jobs=-1)
harga_wajar_oof = np.clip(harga_wajar_oof, a_min=1_000_000, a_max=None)

df_model['harga_wajar_oof'] = harga_wajar_oof
df_model['deviasi_persen'] = ((df_model[target_harga] - df_model['harga_wajar_oof']) / df_model['harga_wajar_oof']) * 100

print(f"Rentang deviasi harga: {df_model['deviasi_persen'].min():.1f}% s/d {df_model['deviasi_persen'].max():.1f}%")

# Fit ulang model regresi harga di SELURUH data (dipakai sbg info "harga
# wajar" di aplikasi akhir -- terpisah dari proses pelabelan di atas)
pipeline_harga_final = Pipeline(steps=[
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
])
pipeline_harga_final.fit(X, y_harga)
with open("model_regresi_harga.pkl", "wb") as f:
    pickle.dump(pipeline_harga_final, f)
print("[INFO] Model regresi harga (referensi) disimpan ke 'model_regresi_harga.pkl'")

# ============================================================
# TAHAP 2: PEMBENTUKAN LABEL LIKUIDITAS (TERTILE, DATA-DRIVEN)
# ============================================================
print("\n" + "=" * 75)
print("TAHAP 2: PEMBENTUKAN LABEL PROXY LIKUIDITAS (TERTILE)")
print("=" * 75)

q33 = df_model['deviasi_persen'].quantile(0.33)
q67 = df_model['deviasi_persen'].quantile(0.67)
print(f"Ambang tertile deviasi harga -> Q33: {q33:.2f}% | Q67: {q67:.2f}%")

def label_dasar(dev):
    if dev <= q33:
        return "Cepat"
    elif dev <= q67:
        return "Sedang"
    else:
        return "Lambat"

df_model['status_likuiditas'] = df_model['deviasi_persen'].apply(label_dasar)

mask_turun = (
    (df_model['status_likuiditas'] == "Cepat") &
    (df_model['km_per_tahun'] > 30000) &
    (df_model['ada_garansi'] == 'Tidak')
)
df_model.loc[mask_turun, 'status_likuiditas'] = "Sedang"

print("Distribusi label proxy:")
print(df_model['status_likuiditas'].value_counts())

# ============================================================
# TAHAP 3: TRAINING CLASSIFIER SUNGGUHAN UNTUK MEMPREDIKSI LABEL
# ============================================================
print("\n" + "=" * 75)
print("TAHAP 3: PELATIHAN & EVALUASI CLASSIFIER LIKUIDITAS")
print("=" * 75)

X_cls = df_model[fitur]
y_cls = df_model['status_likuiditas']

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
}

hasil_metrik = []
trained_pipelines = {}
best_model_name, best_model_pipeline, best_f1 = None, None, -np.inf
label_order = ["Cepat", "Sedang", "Lambat"]

for nama_model, clf in classifiers.items():
    pipeline = Pipeline(steps=[('prep', preprocessor), ('clf', clf)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    trained_pipelines[nama_model] = (pipeline, y_pred)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\nModel: {nama_model}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  F1-Score (macro)  : {f1_macro:.4f}")
    print(f"  Precision (macro) : {prec_macro:.4f}")
    print(f"  Recall (macro)    : {rec_macro:.4f}")

    hasil_metrik.append({
        "Model": nama_model, "Accuracy": acc, "F1_Macro": f1_macro,
        "Precision_Macro": prec_macro, "Recall_Macro": rec_macro
    })

    if f1_macro > best_f1:
        best_f1, best_model_name, best_model_pipeline = f1_macro, nama_model, pipeline

print("\n" + "=" * 75)
print(f"Model Classifier Terbaik Terpilih: {best_model_name} (F1-Macro: {best_f1:.4f})")
print("=" * 75)
print("\nClassification Report (model terbaik):")
_, y_pred_best = trained_pipelines[best_model_name]
print(classification_report(y_test, y_pred_best, labels=label_order))

df_metrik = pd.DataFrame(hasil_metrik)

with open("model_klasifikasi_likuiditas.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)
print(f"[SUKSES] Model klasifikasi likuiditas tersimpan ke 'model_klasifikasi_likuiditas.pkl'")

# ============================================================
# VISUALISASI EVALUASI
# ============================================================
print(f"\nMenghasilkan grafik visualisasi ke folder '{OUTPUT_DIR}/'...")

# --- 1. Komparasi Accuracy & F1-Macro Semua Model ---
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_metrik))
lebar = 0.35
ax.bar(x - lebar/2, df_metrik["Accuracy"], lebar, label="Accuracy", color="#2563EB")
ax.bar(x + lebar/2, df_metrik["F1_Macro"], lebar, label="F1-Macro", color="#059669")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(df_metrik["Model"], rotation=15, ha="right", fontsize=10, fontweight="bold")
ax.set_title("Komparasi Performa Algoritma Klasifikasi Likuiditas", fontsize=13, pad=12)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_metrik_klasifikasi.png"), dpi=300)
plt.close(fig)

# --- 2. Confusion Matrix (Model Terbaik) ---
cm = confusion_matrix(y_test, y_pred_best, labels=label_order)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(label_order)))
ax.set_yticks(range(len(label_order)))
ax.set_xticklabels(label_order)
ax.set_yticklabels(label_order)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual (label proxy)")
ax.set_title(f"Confusion Matrix — {best_model_name}", fontsize=13, pad=10)
for i in range(len(label_order)):
    for j in range(len(label_order)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix.png"), dpi=300)
plt.close(fig)

# --- 3. Distribusi Kelas Label Proxy ---
fig, ax = plt.subplots(figsize=(7, 5))
counts = df_model['status_likuiditas'].value_counts().reindex(label_order)
ax.bar(counts.index, counts.values, color=["#059669", "#F59E0B", "#DC2626"])
ax.set_ylabel("Jumlah Unit")
ax.set_title("Distribusi Label Proxy Likuiditas (Hasil Tertile)", fontsize=13, pad=10)
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_distribusi_label_proxy.png"), dpi=300)
plt.close(fig)

# --- 4. Feature Importance Top 15 (jika tersedia) ---
clf_final = best_model_pipeline.named_steps['clf']
if hasattr(clf_final, "feature_importances_"):
    importances = clf_final.feature_importances_
    nama_fitur = best_model_pipeline.named_steps['prep'].get_feature_names_out()
    df_imp = pd.DataFrame({"Fitur": nama_fitur, "Importance": importances}).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_imp["Fitur"][::-1], df_imp["Importance"][::-1], color="#0D9488")
    ax.set_xlabel("Skor Kepentingan Relatif (Feature Importance)", fontsize=11)
    ax.set_title(f"15 Fitur Paling Berpengaruh — {best_model_name}", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance_klasifikasi.png"), dpi=300)
    plt.close(fig)

# --- 5. [BARU] Akurasi Model Dipecah per Tingkat Kecukupan Sampel ---
if 'jumlah_sampel_kategori' in df_model.columns:
    idx_test = X_test.index
    sampel_test = df_model.loc[idx_test, 'jumlah_sampel_kategori']
    benar = (y_test.values == y_pred_best)
    grup_rendah = sampel_test < AMBANG_KEYAKINAN_SAMPEL
    akurasi_rendah = benar[grup_rendah.values].mean() if grup_rendah.sum() > 0 else np.nan
    akurasi_cukup = benar[~grup_rendah.values].mean() if (~grup_rendah).sum() > 0 else np.nan

    print(f"\n[DIAGNOSTIK] Akurasi pada data uji, dipecah per kecukupan sampel:")
    print(f"  Sampel kategori < {AMBANG_KEYAKINAN_SAMPEL} (keyakinan rendah) : {akurasi_rendah:.2%} (n={int(grup_rendah.sum())})")
    print(f"  Sampel kategori >= {AMBANG_KEYAKINAN_SAMPEL} (keyakinan cukup) : {akurasi_cukup:.2%} (n={int((~grup_rendah).sum())})")

    fig, ax = plt.subplots(figsize=(6, 5))
    label_grup = [f"Sampel < {AMBANG_KEYAKINAN_SAMPEL}\n(keyakinan rendah)", f"Sampel >= {AMBANG_KEYAKINAN_SAMPEL}\n(keyakinan cukup)"]
    nilai_grup = [akurasi_rendah, akurasi_cukup]
    warna = ["#DC2626", "#059669"]
    ax.bar(label_grup, nilai_grup, color=warna)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Akurasi")
    ax.set_title("Akurasi Classifier: Kategori Minim Sampel vs Cukup Sampel", fontsize=12, pad=10)
    for i, v in enumerate(nilai_grup):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_akurasi_per_kecukupan_sampel.png"), dpi=300)
    plt.close(fig)

df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik_klasifikasi.csv"), index=False)
print(f"[SELESAI] Model dan seluruh visualisasi klasifikasi berhasil disimpan ke folder '{OUTPUT_DIR}/'.")