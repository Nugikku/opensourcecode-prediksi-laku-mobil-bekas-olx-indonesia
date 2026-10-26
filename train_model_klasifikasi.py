"""
AutoLaku: Pelatihan & Evaluasi Model Klasifikasi Kecepatan Laku
Output disesuaikan dengan standar Materi P12 (Model Evaluation) & P13 (Model Comparison & Selection).
"""

import os, json, pickle, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, mean_absolute_percentage_error,
                             mean_squared_error, mean_absolute_error, root_mean_squared_error,
                             roc_curve, auc)

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

df_model = df.dropna(subset=fitur_harga + ['harga']).copy()
print(f"Total sampel dataset: {len(df_model):,} baris")

def buat_prep(num_cols):
    return ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                              ('num', StandardScaler(), num_cols)])

def buat_regresor():
    reg = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
        func=np.log1p, inverse_func=np.expm1)
    return Pipeline([('prep', buat_prep(numerical_cols)), ('reg', reg)])

# ============================================================
# TAHAP 1: PREDIKSI HARGA WAJAR (REGRESI)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 1: PREDIKSI HARGA WAJAR (Out-of-Fold ML + Fallback Usia)\n" + "=" * 75)
oof = cross_val_predict(buat_regresor(), df_model[fitur_harga], df_model['harga'],
                        cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1)
df_model['harga_wajar_model'] = np.clip(oof, 1_000_000, None)

sampel_lain = df_model['jumlah_sampel_kategori'] - 1
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

mape_final = mean_absolute_percentage_error(df_model['harga'], df_model['harga_wajar_final']) * 100
mae_final = mean_absolute_error(df_model['harga'], df_model['harga_wajar_final'])
rmse_final = root_mean_squared_error(df_model['harga'], df_model['harga_wajar_final'])

regresi_final = buat_regresor().fit(df_model[fitur_harga], df_model['harga'])
with open("model_regresi_harga.pkl", "wb") as f:
    pickle.dump(regresi_final, f)

# ============================================================
# TAHAP 2: FORMULASI LABEL PROXY (TERTILE)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 2: PEMBENTUKAN LABEL PROXY LIKUIDITAS\n" + "=" * 75)
df_model['deviasi_persen'] = ((df_model['harga'] - df_model['harga_wajar_final'])
                              / df_model['harga_wajar_final'] * 100).clip(DEVIASI_MIN, DEVIASI_MAX)
q33, q67 = df_model['deviasi_persen'].quantile([0.33, 0.67])

df_model['status_likuiditas'] = np.where(df_model['deviasi_persen'] <= q33, "Cepat",
                                np.where(df_model['deviasi_persen'] <= q67, "Sedang", "Lambat"))
turun = ((df_model['status_likuiditas'] == "Cepat") & (df_model['km_per_tahun'] > 30000)
         & (df_model['ada_garansi'] == 'Tidak'))
df_model.loc[turun, 'status_likuiditas'] = "Sedang"

# ============================================================
# TAHAP 3: PERBANDINGAN MODEL & CROSS-VALIDATION (P12 & P13)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 3: EVALUASI & KOMPARASI MODEL (P12 & P13 STANDARDS)\n" + "=" * 75)
X, y = df_model[fitur_kls], df_model['status_likuiditas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

kandidat_model = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
}

hasil_komparasi = []
pipes = {}

for nama, clf in kandidat_model.items():
    pipe = Pipeline([('prep', buat_prep(numerical_cols + ['deviasi_persen'])), ('clf', clf)])
    pipe.fit(X_train, y_train)
    pipes[nama] = pipe
    
    # Stratified 5-Fold Cross Validation (P12/P13)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1_macro')
    
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    hasil_komparasi.append({
        "Model": nama,
        "Accuracy": round(acc, 4),
        "Precision_Macro": round(prec, 4),
        "Recall_Macro": round(rec, 4),
        "F1_Macro": round(f1, 4),
        "CV_F1_Mean": round(cv_scores.mean(), 4),
        "CV_F1_Std": round(cv_scores.std(), 4)
    })

df_komparasi = pd.DataFrame(hasil_komparasi)
print(df_komparasi.to_string(index=False))

# Pilih Model Terbaik Berdasarkan CV F1-Mean
best_model_name = df_komparasi.sort_values("CV_F1_Mean", ascending=False).iloc[0]["Model"]
best_pipe = pipes[best_model_name]
print(f"\nModel Terbaik Pilihan: {best_model_name}")

# Simpan Model Terbaik
with open("model_klasifikasi_likuiditas.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

# ============================================================
# TAHAP 4: GENERASI GRAFIK OUTPUT (STANDAR P12 & P13)
# ============================================================
print("\n" + "=" * 75 + "\nTAHAP 4: GENERASI GRAFIK STANDAR P12 & P13\n" + "=" * 75)

# 1. Grafik Komparasi Metrik (P13 Slide 11-15)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_komparasi))
width = 0.2
ax.bar(x - width*1.5, df_komparasi["Accuracy"], width, label="Accuracy", color="#2563EB")
ax.bar(x - width*0.5, df_komparasi["Precision_Macro"], width, label="Precision", color="#D97706")
ax.bar(x + width*0.5, df_komparasi["Recall_Macro"], width, label="Recall", color="#DC2626")
ax.bar(x + width*1.5, df_komparasi["F1_Macro"], width, label="F1-Score", color="#059669")
ax.set_xticks(x); ax.set_xticklabels(df_komparasi["Model"], rotation=10)
ax.set_ylim(0, 1.1); ax.set_title("Perbandingan Metrik Evaluasi Model (P13)", fontsize=12, pad=10)
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_metrik_klasifikasi.png"), dpi=300); plt.close()

# 2. Confusion Matrix Model Terbaik (P12 Slide 9-10)
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=LABEL_ORDER)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(LABEL_ORDER); ax.set_yticklabels(LABEL_ORDER)
ax.set_xlabel("Prediksi Model"); ax.set_ylabel("Kelas Aktual")
ax.set_title(f"Confusion Matrix - {best_model_name} (P12)", fontsize=12, pad=10)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix.png"), dpi=300); plt.close()

# 3. Grafik Stabilitas Cross-Validation Score (P13 Slide 25-26)
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(df_komparasi["Model"], df_komparasi["CV_F1_Mean"], yerr=df_komparasi["CV_F1_Std"], 
            fmt='o', color='#0284C7', ecolor='#EF4444', elinewidth=2, capsize=5)
ax.set_ylabel("Mean F1-Score"); ax.set_title("Stabilitas Model via 5-Fold Cross-Validation (P13)", fontsize=12, pad=10)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "3_stabilitas_cross_validation.png"), dpi=300); plt.close()

# 4. Feature Importance (P13 Slide 39)
clf_final = best_pipe.named_steps['clf']
if hasattr(clf_final, "feature_importances_"):
    imp = pd.DataFrame({"Fitur": best_pipe.named_steps['prep'].get_feature_names_out(),
                        "Importance": clf_final.feature_importances_}).sort_values("Importance", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp["Fitur"][::-1], imp["Importance"][::-1], color="#0D9488")
    ax.set_title(f"Feature Importance (Explainability) - {best_model_name} (P13)", fontsize=12, pad=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance.png"), dpi=300); plt.close()

# Simpan Ringkasan CSV
df_komparasi.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik_klasifikasi.csv"), index=False)

meta = {
    "q33": float(q33), "q67": float(q67), "label_order": LABEL_ORDER,
    "model_klasifikasi": best_model_name,
    "mape_harga_wajar_pct": round(mape_final, 2),
    "mae_harga_wajar_rupiah": round(mae_final, 0),
    "rmse_harga_wajar_rupiah": round(rmse_final, 0),
    "tanggal_latih": datetime.now().strftime("%Y-%m-%d")
}
json.dump(meta, open("autolaku_meta.json", "w"), indent=2)

print(f"[SUKSES] Seluruh grafik dan ringkasan metrik sesuai standar P12 & P13 disimpan di folder '{OUTPUT_DIR}/'.")