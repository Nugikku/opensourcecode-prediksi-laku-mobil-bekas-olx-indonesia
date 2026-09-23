import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

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
numerical_cols = ['harga', 'tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun', 'deviasi_harga_pasar']
fitur = categorical_cols + numerical_cols
target = 'kategori_penjualan'

df_model = df.dropna(subset=fitur + [target]).copy()
y = df_model[target]
X = df_model[fitur]

kelas_urut = ['Cepat', 'Sedang', 'Lambat']
daftar_kelas = [k for k in kelas_urut if k in y.unique()]

print(f"Total baris data siap training : {len(df_model)}")
print(f"Distribusi Target             : {y.value_counts().to_dict()}")

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
trained_pipelines = {}
best_model_name, best_model_pipeline, best_f1 = None, None, -1.0

print("\n" + "=" * 65)
print("PELATIHAN & EVALUASI 5 ALGORITMA KLASIFIKASI")
print("=" * 65)

for nama_model, classifier in models.items():
    pipeline = Pipeline(steps=[('prep', preprocessor), ('clf', classifier)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    trained_pipelines[nama_model] = (pipeline, y_pred)
    
    auc_score = 0.5
    if hasattr(pipeline.named_steps['clf'], "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
            auc_score = roc_auc_score(
                y_test, 
                y_prob, 
                labels=pipeline.named_steps['clf'].classes_, 
                multi_class="ovr", 
                average="weighted"
            )
        except Exception:
            auc_score = 0.5

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nModel: {nama_model}")
    print(f"  Akurasi   : {acc * 100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc_score:.4f}")

    hasil_metrik.append({
        "Model": nama_model,
        "Akurasi": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc_score
    })

    if f1 > best_f1:
        best_f1, best_model_name, best_model_pipeline = f1, nama_model, pipeline

print("\n" + "=" * 65)
print(f"Model Terbaik Terpilih: {best_model_name} (F1-Score: {best_f1:.4f})")
print("=" * 65)

df_metrik = pd.DataFrame(hasil_metrik)

with open("model_klasifikasi_penjualan.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)
print(f"[SUKSES] Model tersimpan ke 'model_klasifikasi_penjualan.pkl'")

# ============================================================
# PEMBUATAN 7 GRAFIK VISUALISASI DETAIL
# ============================================================
print(f"\nMenghasilkan 7 grafik analisis visual ke folder '{OUTPUT_DIR}/'...")

# --- 1. Komparasi Semua Metrik Performa ---
fig, ax = plt.subplots(figsize=(12, 6))
posisi = np.arange(len(df_metrik))
lebar = 0.16
warna = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]
kolom_eval = ["Akurasi", "Precision", "Recall", "F1-Score", "ROC-AUC"]

for i, kol in enumerate(kolom_eval):
    ax.bar(posisi + (i * lebar) - (2 * lebar), df_metrik[kol], lebar, label=kol, color=warna[i])

ax.set_xticks(posisi)
ax.set_xticklabels(df_metrik["Model"], fontsize=11, fontweight="bold")
ax.set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11)
ax.set_ylim(0, 1.22)
ax.set_title("Perbandingan Komprehensif 5 Metrik Evaluasi Model Klasifikasi", fontsize=14, pad=12)
ax.legend(loc="upper right", frameon=True, ncol=5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_semua_metrik.png"), dpi=300)
plt.close(fig)

# --- 2. Matriks Konfusi 5 Model Berdampingan ---
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for idx, (m_name, (pipe, preds)) in enumerate(trained_pipelines.items()):
    cm = confusion_matrix(y_test, preds, labels=daftar_kelas)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=daftar_kelas)
    disp.plot(cmap="Blues", ax=axes[idx], colorbar=False)
    axes[idx].set_title(m_name, fontsize=12, fontweight="bold")
    axes[idx].grid(False)
plt.suptitle("Perbandingan Confusion Matrix Semua Algoritma (Ground Truth vs Prediksi)", fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix_komparasi.png"), dpi=300)
plt.close(fig)

# --- 3. Kurva ROC Multi-Class (One-vs-Rest) Model Terbaik ---
if hasattr(best_model_pipeline.named_steps['clf'], "predict_proba"):
    y_test_bin = label_binarize(y_test, classes=daftar_kelas)
    y_prob_best = best_model_pipeline.predict_proba(X_test)
    classes_clf = list(best_model_pipeline.named_steps['clf'].classes_)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    warna_kelas = ["#10B981", "#F59E0B", "#EF4444"]
    
    for i, nama_kls in enumerate(daftar_kelas):
        if nama_kls in classes_clf:
            idx_clf = classes_clf.index(nama_kls)
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_best[:, idx_clf])
            skor_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=warna_kelas[i % len(warna_kelas)], lw=2, label=f"Kelas {nama_kls} (AUC = {skor_auc:.2f})")
    
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Garis Acak (Baseline)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Kurva ROC Multi-Class (One-vs-Rest) — {best_model_name}", fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_kurva_roc_multiclass.png"), dpi=300)
    plt.close(fig)

# --- 4. Feature Importance (Peringkat Fitur Berpengaruh) ---
clf = best_model_pipeline.named_steps['clf']
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    nama_fitur = best_model_pipeline.named_steps['prep'].get_feature_names_out()
    df_imp = pd.DataFrame({"Fitur": nama_fitur, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_imp["Fitur"][::-1], df_imp["Importance"][::-1], color="#6366F1")
    ax.set_xlabel("Skor Kepentingan (Relative Importance)", fontsize=11)
    ax.set_title(f"15 Fitur Paling Berpengaruh dalam Penentuan Likuiditas — {best_model_name}", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance.png"), dpi=300)
    plt.close(fig)

# --- 5. Distribusi Target Kelas Likuiditas ---
fig, ax = plt.subplots(figsize=(7, 5))
hitung_kelas = df_model[target].value_counts().reindex(daftar_kelas)
warna_dist = ["#10B981", "#F59E0B", "#EF4444"]
batang = ax.bar(hitung_kelas.index, hitung_kelas.values, color=warna_dist, width=0.5)

for b in batang:
    tinggi = b.get_height()
    persen = (tinggi / len(df_model)) * 100
    ax.text(b.get_x() + b.get_width() / 2, tinggi + 15, f"{tinggi:,}\n({persen:.1f}%)", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("Jumlah Unit Mobil", fontsize=11)
ax.set_ylim(0, hitung_kelas.max() * 1.18)
ax.set_title("Proporsi Sampel Target Likuiditas Penjualan (Ground Truth)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_distribusi_target_kelas.png"), dpi=300)
plt.close(fig)

# --- 6. Boxplot Hubungan Deviasi Harga Pasar & Likuiditas ---
fig, ax = plt.subplots(figsize=(8, 5))
data_box = [df_model[df_model[target] == k]['deviasi_harga_pasar'].dropna().values for k in daftar_kelas]
bp = ax.boxplot(data_box, patch_artist=True, tick_labels=daftar_kelas, showfliers=False)

for patch, w in zip(bp['boxes'], warna_dist):
    patch.set_facecolor(w)
    patch.set_alpha(0.7)

ax.axhline(0, color="#6B7280", linestyle="--", linewidth=1.2, label="Harga Median Pasar")
ax.set_ylabel("Deviasi terhadap Harga Median Pasar (%)", fontsize=11)
ax.set_title("Korelasi Deviasi Harga Penawaran terhadap Kategori Likuiditas", fontsize=13, pad=12)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_analisis_harga_vs_likuiditas.png"), dpi=300)
plt.close(fig)

# --- 7. Radar Chart Keseimbangan Metrik Model Terbaik ---
baris_best = df_metrik[df_metrik["Model"] == best_model_name].iloc[0]
label_radar = ["Akurasi", "Precision", "Recall", "F1-Score", "ROC-AUC"]
nilai_radar = [baris_best[m] for m in label_radar]
nilai_radar += nilai_radar[:1]

sudut = np.linspace(0, 2 * np.pi, len(label_radar), endpoint=False).tolist()
sudut += sudut[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(sudut, nilai_radar, color="#2563EB", linewidth=2)
ax.fill(sudut, nilai_radar, color="#3B82F6", alpha=0.3)
ax.set_xticks(sudut[:-1])
ax.set_xticklabels(label_radar, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_title(f"Profil Keseimbangan Metrik Evaluasi ({best_model_name})", fontsize=13, y=1.08)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_radar_chart_performa.png"), dpi=300)
plt.close(fig)

# Simpan Ringkasan CSV
df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik.csv"), index=False)
print(f"\n[SELESAI] Seluruh 7 grafik dan ringkasan metrik berhasil diekspor ke folder '{OUTPUT_DIR}/'.")