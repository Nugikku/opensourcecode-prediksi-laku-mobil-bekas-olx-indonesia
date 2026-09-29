"""
AutoLaku: Pelatihan Model Klasifikasi Kecepatan Laku (Cepat/Sedang/Lambat)
Arsitektur: Harga Wajar (regresi log-harga + fallback usia) -> Deviasi Harga
            -> Label Proxy (tertile) -> Classifier ML

Catatan metodologi:
  Data "terjual" tidak tersedia (difilter OLX), sehingga label bersifat PROXY:
  1. Harga wajar tiap iklan dihitung out-of-fold (tanpa membaca iklan itu
     sendiri). Kategori bersampel sedikit memakai regresi usia leave-one-out.
  2. Deviasi harga iklan terhadap harga wajar dibagi 3 kelas dgn TERTILE
     (persentil 33/67), bukan ambang buatan.
  3. Unit berkilometer sangat tinggi & tanpa garansi diturunkan dari Cepat
     ke Sedang.
  4. Classifier dilatih dgn fitur unit + deviasi harga, sehingga vonis
     BERUBAH mengikuti rencana harga yang diketik penjual.
  Karena label dibentuk dari deviasi harga, akurasi tinggi TIDAK membuktikan
  vonis cocok dengan kenyataan. Validasi kasar: umur iklan (tanggal_posting).
  Pemilihan model dibatasi pada model yang MONOTON: harga naik tidak boleh
  membuat vonis jadi lebih cepat laku.
"""

import os, json, pickle, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, mean_absolute_percentage_error)

from processing import AMBANG_KEYAKINAN_SAMPEL, hitung_harga_wajar_fallback, klip_deviasi, DEVIASI_MIN, DEVIASI_MAX

warnings.filterwarnings("ignore")
OUTPUT_DIR = f"output_klasifikasi_{datetime.now().strftime('%d-%m-%Y')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

file_dataset = "dataset_olx_bersih.csv"
if not os.path.exists(file_dataset):
    raise FileNotFoundError(f"File '{file_dataset}' tidak ditemukan. Jalankan processing.py terlebih dahulu.")
df = pd.read_csv(file_dataset)

categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
numerical_cols = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']
fitur_harga = categorical_cols + numerical_cols
fitur_kls = fitur_harga + ['deviasi_persen']
LABEL_ORDER = ["Cepat", "Sedang", "Lambat"]
RANK = {l: i for i, l in enumerate(LABEL_ORDER)}

df_model = df.dropna(subset=fitur_harga + ['harga']).copy()   # index asli dipertahankan
print(f"Total sampel: {len(df_model):,} baris")

def buat_prep(num_cols):
    return ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                              ('num', StandardScaler(), num_cols)])

def buat_regresor():
    reg = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
        func=np.log1p, inverse_func=np.expm1)
    return Pipeline([('prep', buat_prep(numerical_cols)), ('reg', reg)])

# ============================================================
# TAHAP 1: HARGA WAJAR (OOF + fallback LOO utk sampel sedikit)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 1: HARGA WAJAR (out-of-fold, log-harga, fallback usia)\n" + "=" * 75)
oof = cross_val_predict(buat_regresor(), df_model[fitur_harga], df_model['harga'],
                        cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1)
df_model['harga_wajar_model'] = np.clip(oof, 1_000_000, None)

sampel_lain = df_model['jumlah_sampel_kategori'] - 1        # tidak menghitung baris itu sendiri
final, sumber = [], []
for idx, r in df_model.iterrows():
    hw, sm = float(r['harga_wajar_model']), "model ML"
    if sampel_lain[idx] < AMBANG_KEYAKINAN_SAMPEL:
        hf, s = hitung_harga_wajar_fallback(df_model, r['merek'], r['model'], r['usia_mobil'], exclude_idx=idx)
        if hf is not None:
            hw, sm = hf, s
    final.append(hw); sumber.append(sm)
df_model['harga_wajar_final'] = final
df_model['sumber_harga_wajar'] = sumber

mape_model = mean_absolute_percentage_error(df_model['harga'], df_model['harga_wajar_model']) * 100
mape_final = mean_absolute_percentage_error(df_model['harga'], df_model['harga_wajar_final']) * 100
print(f"Galat rata-rata (MAPE) harga wajar: hanya model ML {mape_model:.1f}% -> model+fallback {mape_final:.1f}%")
print(df_model['sumber_harga_wajar'].value_counts().to_string())

regresi_final = buat_regresor().fit(df_model[fitur_harga], df_model['harga'])
with open("model_regresi_harga.pkl", "wb") as f:
    pickle.dump(regresi_final, f)

# ============================================================
# TAHAP 2: LABEL PROXY (TERTILE)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 2: LABEL PROXY LIKUIDITAS (TERTILE)\n" + "=" * 75)
df_model['deviasi_persen'] = ((df_model['harga'] - df_model['harga_wajar_final'])
                              / df_model['harga_wajar_final'] * 100).clip(DEVIASI_MIN, DEVIASI_MAX)
q33, q67 = df_model['deviasi_persen'].quantile([0.33, 0.67])
print(f"Ambang tertile deviasi -> Q33: {q33:.2f}% | Q67: {q67:.2f}%")

df_model['status_likuiditas'] = np.where(df_model['deviasi_persen'] <= q33, "Cepat",
                                np.where(df_model['deviasi_persen'] <= q67, "Sedang", "Lambat"))
turun = ((df_model['status_likuiditas'] == "Cepat") & (df_model['km_per_tahun'] > 30000)
         & (df_model['ada_garansi'] == 'Tidak'))
df_model.loc[turun, 'status_likuiditas'] = "Sedang"
print(df_model['status_likuiditas'].value_counts().to_string())

# ============================================================
# TAHAP 3: CLASSIFIER (dengan uji monotonisitas terhadap harga)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 3: PELATIHAN & EVALUASI CLASSIFIER\n" + "=" * 75)
X, y = df_model[fitur_kls], df_model['status_likuiditas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def pelanggaran_monoton(pipe, X_ref, n=300):
    """Persentase unit yang vonisnya menjadi LEBIH CEPAT saat harga dinaikkan."""
    sampel = X_ref.sample(min(n, len(X_ref)), random_state=1)
    grid = np.linspace(-80, 250, 34)
    besar = sampel.loc[sampel.index.repeat(len(grid))].copy()
    besar['deviasi_persen'] = np.tile(grid, len(sampel))
    rank = np.array([RANK[p] for p in pipe.predict(besar)]).reshape(len(sampel), len(grid))
    return float((np.diff(rank, axis=1) < 0).any(axis=1).mean() * 100)

kandidat = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
}
def rerata_keyakinan_puncak(pipe, X_ref):
    """Rata-rata probabilitas kelas yang dipilih model (mendekati 1.0 terus-menerus
    = tanda model overconfident, mis. Decision Tree tunggal yang tiap daunnya
    murni satu kelas -> keyakinan 100% padahal belum tentu benar)."""
    clf = pipe.named_steps['clf']
    if not hasattr(clf, "predict_proba"):
        return np.nan
    return float(pipe.predict_proba(X_ref).max(axis=1).mean())

hasil, pipes = [], {}
for nama, clf in kandidat.items():
    p = Pipeline([('prep', buat_prep(numerical_cols + ['deviasi_persen'])), ('clf', clf)]).fit(X_train, y_train)
    yp = p.predict(X_test)
    langgar = pelanggaran_monoton(p, X_test)
    keyakinan = rerata_keyakinan_puncak(p, X_test)
    pipes[nama] = (p, yp)
    hasil.append({"Model": nama, "Accuracy": accuracy_score(y_test, yp),
                  "F1_Macro": f1_score(y_test, yp, average='macro'),
                  "Precision_Macro": precision_score(y_test, yp, average='macro', zero_division=0),
                  "Recall_Macro": recall_score(y_test, yp, average='macro', zero_division=0),
                  "Pelanggaran_Monoton_%": langgar, "Rerata_Keyakinan_Puncak": keyakinan})
    print(f"{nama:20s} acc {hasil[-1]['Accuracy']:.4f} | F1 {hasil[-1]['F1_Macro']:.4f} | "
          f"pelanggaran monoton {langgar:.1f}% | rerata keyakinan puncak {keyakinan:.3f}")

df_metrik = pd.DataFrame(hasil)

# Kriteria pemilihan model, berurutan:
#  1) HARUS monoton terhadap harga (0% pelanggaran) -- syarat mutlak.
#  2) HARUS tidak overconfident: model tunggal (Decision Tree) sering
#     menghasilkan probabilitas 0%/100% karena satu daun murni satu kelas --
#     itu bukan bukti "sangat yakin", hanya artefak algoritma. Model
#     ensemble (Random Forest/Gradient Boosting) & Logistic Regression
#     merata-ratakan banyak estimator/kelas sehingga keyakinannya lebih wajar.
#     Ambang > 0.97 dianggap overconfident & dihindari kalau ada alternatif.
#  3) Di antara yang lolos (1) & (2), pilih F1-macro tertinggi.
monoton = df_metrik[df_metrik["Pelanggaran_Monoton_%"] == 0]
if monoton.empty:
    monoton = df_metrik.sort_values("Pelanggaran_Monoton_%").head(1)
    print("\n[PERINGATAN] Tidak ada model yang 100% monoton; dipilih yang paling mendekati.")

wajar = monoton[monoton["Rerata_Keyakinan_Puncak"] <= 0.97]
pool = wajar if len(wajar) else monoton
if wajar.empty and len(monoton) > 1:
    print("[INFO] Semua model monoton overconfident (keyakinan puncak > 0.97); "
          "dipilih F1-macro terbaik dari seluruh model monoton apa adanya.")

best_name = pool.sort_values("F1_Macro", ascending=False).iloc[0]["Model"]
best_pipe, y_pred_best = pipes[best_name]
print(f"\nModel terpilih: {best_name} (F1-macro terbaik di antara model yang monoton & tidak overconfident)")
print(classification_report(y_test, y_pred_best, labels=LABEL_ORDER))

with open("model_klasifikasi_likuiditas.pkl", "wb") as f:
    pickle.dump(best_pipe, f)
baris_terpilih = df_metrik[df_metrik["Model"] == best_name].iloc[0]
meta = {"q33": float(q33), "q67": float(q67), "label_order": LABEL_ORDER, "model_klasifikasi": best_name,
        "rerata_keyakinan_puncak": round(float(baris_terpilih["Rerata_Keyakinan_Puncak"]), 4),
        "fitur_harga": fitur_harga, "fitur_klasifikasi": fitur_kls, "ambang_sampel": AMBANG_KEYAKINAN_SAMPEL,
        "mape_harga_wajar_pct": round(mape_final, 2), "tanggal_latih": datetime.now().strftime("%Y-%m-%d")}
json.dump(meta, open("autolaku_meta.json", "w"), indent=2)
print("[SUKSES] Tersimpan: model_regresi_harga.pkl, model_klasifikasi_likuiditas.pkl, autolaku_meta.json")

# ============================================================
# VALIDASI KASAR LABEL PROXY: umur iklan & status terjual
# ============================================================
print("\n" + "=" * 75 + "\nVALIDASI KASAR LABEL PROXY\n" + "=" * 75)
waktu = pd.to_datetime(df_model['tanggal_posting'], errors='coerce', utc=True)
df_model['umur_iklan_hari'] = (waktu.max() - waktu).dt.total_seconds() / 86400
ok = df_model['umur_iklan_hari'].notna()
rho, pval = spearmanr(df_model.loc[ok, 'deviasi_persen'], df_model.loc[ok, 'umur_iklan_hari'])
rata_umur = df_model[ok].groupby('status_likuiditas')['umur_iklan_hari'].mean().reindex(LABEL_ORDER)
print(f"Korelasi Spearman deviasi harga vs umur iklan: rho={rho:.3f} (p={pval:.3g})")
print("Rata-rata umur iklan (hari) per label:\n" + rata_umur.round(1).to_string())
print("Interpretasi: label 'Lambat' seharusnya beriklan lebih lama. Umur iklan hanyalah pendekatan "
      "(iklan bisa di-bump), bukan bukti terjual.")
n_terjual = int((df_model['status_iklan_terjual'] == 'Terjual').sum()) if 'status_iklan_terjual' in df_model else 0
print(f"Iklan bertanda 'Terjual' di dataset: {n_terjual} baris -> terlalu sedikit dipakai sebagai validasi.")

# ============================================================
# VISUALISASI
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(df_metrik))
ax.bar(x - .175, df_metrik["Accuracy"], .35, label="Accuracy", color="#2563EB")
ax.bar(x + .175, df_metrik["F1_Macro"], .35, label="F1-Macro", color="#059669")
ax.set_xticks(x); ax.set_xticklabels(df_metrik["Model"], rotation=15, ha="right"); ax.set_ylim(0, 1.05)
ax.set_title("Komparasi Algoritma Klasifikasi AutoLaku"); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_metrik_klasifikasi.png"), dpi=300); plt.close(fig)

cm = confusion_matrix(y_test, y_pred_best, labels=LABEL_ORDER)
fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(LABEL_ORDER); ax.set_yticklabels(LABEL_ORDER)
ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual (label proxy)"); ax.set_title(f"Confusion Matrix - {best_name}")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix.png"), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5)); cnt = df_model['status_likuiditas'].value_counts().reindex(LABEL_ORDER)
ax.bar(cnt.index, cnt.values, color=["#059669", "#F59E0B", "#DC2626"]); ax.set_title("Distribusi Label Proxy (Tertile)")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "3_distribusi_label_proxy.png"), dpi=300); plt.close(fig)

clf_final = best_pipe.named_steps['clf']
if hasattr(clf_final, "feature_importances_"):
    imp = pd.DataFrame({"Fitur": best_pipe.named_steps['prep'].get_feature_names_out(),
                        "Importance": clf_final.feature_importances_}).sort_values("Importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(imp["Fitur"][::-1], imp["Importance"][::-1], color="#0D9488")
    ax.set_title(f"15 Fitur Terpenting - {best_name}"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance_klasifikasi.png"), dpi=300); plt.close(fig)

sampel_test = df_model.loc[X_test.index, 'jumlah_sampel_kategori'] - 1
benar = (y_test.values == y_pred_best); rendah = (sampel_test < AMBANG_KEYAKINAN_SAMPEL).values
akurasi = [benar[rendah].mean() if rendah.any() else np.nan, benar[~rendah].mean() if (~rendah).any() else np.nan]
print(f"\nAkurasi kategori sampel < {AMBANG_KEYAKINAN_SAMPEL}: {akurasi[0]:.2%} | sampel >= {AMBANG_KEYAKINAN_SAMPEL}: {akurasi[1]:.2%}")
fig, ax = plt.subplots(figsize=(6, 5)); ax.bar([f"< {AMBANG_KEYAKINAN_SAMPEL}\n(keyakinan rendah)", f">= {AMBANG_KEYAKINAN_SAMPEL}\n(cukup)"],
                                               akurasi, color=["#DC2626", "#059669"]); ax.set_ylim(0, 1.05)
ax.set_title("Akurasi per Kecukupan Sampel"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_akurasi_per_kecukupan_sampel.png"), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5)); ax.bar(LABEL_ORDER, rata_umur.values, color=["#059669", "#F59E0B", "#DC2626"])
ax.set_ylabel("Rata-rata umur iklan (hari)"); ax.set_title("Validasi Kasar: Umur Iklan per Label"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_umur_iklan_per_label.png"), dpi=300); plt.close(fig)

df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik_klasifikasi.csv"), index=False)
print(f"[SELESAI] Grafik & ringkasan disimpan di '{OUTPUT_DIR}/'.")