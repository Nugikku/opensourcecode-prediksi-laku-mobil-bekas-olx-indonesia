"""
train_model_klasifikasi.py
--------------------------------------------------------------------------------
Tahap 4: Pelatihan & Evaluasi Model Klasifikasi Likuiditas Mobil Bekas OLX
Standar Perkuliahan Sains Data / Data Mining (P6, P9, P12, P13)
--------------------------------------------------------------------------------
- Bebas Data Leakage (Split Train 80% & Test 20% Terisolasi)
- 6 Algoritma (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting)
- Ablation Study (Skenario A: Tanpa Deviasi Harga vs Skenario B: Dengan Deviasi Harga)
- Evaluasi 5-Fold Stratified Cross-Validation
- Generasi 6 Grafik Visualisasi Akademis dalam Persentase (%)
"""

import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Import 6 Algoritma Perkuliahan (P6, P9, P12, P13)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

# Konfigurasi Direktori & Theme Plotting
OUTPUT_DIR = "output_klasifikasi_terbaru"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

DATASET_FILE = "dataset_olx_berlabel.csv"
META_SELECTION_FILE = "analisis_fitur_meta.json"
LABEL_ORDER = ["Cepat", "Sedang", "Lambat"]


def main():
    print("=" * 85)
    print("   TRAIN_MODEL_KLASIFIKASI.PY - PELATIHAN & EVALUASI 6 GRAFIK MATERI (P6-P13)   ")
    print("=" * 85)

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"File dataset '{DATASET_FILE}' tidak ditemukan.")

    df = pd.read_csv(DATASET_FILE)

    # 1. Memuat Fitur Terpilih dari Metadata Step 3 (analisis_fitur.py)
    if os.path.exists(META_SELECTION_FILE):
        with open(META_SELECTION_FILE, "r") as f:
            meta_sel = json.load(f)
        num_cols = meta_sel["fitur_numerik_terpilih"]
        cat_cols = meta_sel["fitur_kategorik_terpilih"]
    else:
        num_cols = ['usia_mobil', 'jarak_tempuh', 'km_per_tahun', 'indikasi_kredit_tdp']
        cat_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']

    print(f"Fitur Numerik Terpilih   : {num_cols}")
    print(f"Fitur Kategorik Terpilih : {cat_cols}")

    # 2. Data Split Murni (Mencegah Data Leakage)
    df_train = df[df['is_train'] == 1].copy()
    df_test = df[df['is_train'] == 0].copy()

    print(f"\nJumlah Data Training : {len(df_train):,} baris")
    print(f"Jumlah Data Testing  : {len(df_test):,} baris (Terisolasi)")

    # Function Helper Preprocessor
    def buat_prep(fitur_num, fitur_cat):
        return ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), fitur_cat),
            ('num', StandardScaler(), fitur_num)
        ])

    # 3. Definisi 6 Kandidat Algoritma
    kandidat_model = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ============================================================
    # SKENARIO A: ABLATION STUDY (Spesifikasi Murni Tanpa Deviasi Harga)
    # ============================================================
    print("\n" + "-" * 85)
    print("SKENARIO A: ABLATION STUDY (Evaluasi Murni Tanpa Fitur Deviasi Harga)")
    print("-" * 85)

    X_train_a = df_train[num_cols + cat_cols]
    y_train_a = df_train['status_likuiditas']
    X_test_a = df_test[num_cols + cat_cols]
    y_test_a = df_test['status_likuiditas']

    res_skenario_a = []
    for nama, clf in kandidat_model.items():
        pipe = Pipeline([('prep', buat_prep(num_cols, cat_cols)), ('clf', clf)])
        cv_res = cross_validate(pipe, X_train_a, y_train_a, cv=skf, scoring='f1_macro', return_train_score=True)
        pipe.fit(X_train_a, y_train_a)
        y_pred_test = pipe.predict(X_test_a)
        res_skenario_a.append({
            "Model": nama,
            "CV Val F1 (%)": round(cv_res['test_score'].mean() * 100, 2),
            "Test F1 (%)": round(f1_score(y_test_a, y_pred_test, average='macro') * 100, 2)
        })
    df_res_a = pd.DataFrame(res_skenario_a)
    print(df_res_a.to_string(index=False))

    # ============================================================
    # SKENARIO B: MODEL UTAMA APLIKASI (Dengan Fitur Deviasi Harga)
    # ============================================================
    print("\n" + "-" * 85)
    print("SKENARIO B: MODEL UTAMA APLIKASI (Dengan Fitur Deviasi Harga)")
    print("-" * 85)

    num_cols_b = num_cols + ['deviasi_persen']
    X_train_b = df_train[num_cols_b + cat_cols]
    y_train_b = df_train['status_likuiditas']
    X_test_b = df_test[num_cols_b + cat_cols]
    y_test_b = df_test['status_likuiditas']

    res_skenario_b = []
    pipes_b = {}

    for nama, clf in kandidat_model.items():
        pipe = Pipeline([('prep', buat_prep(num_cols_b, cat_cols)), ('clf', clf)])
        cv_res = cross_validate(pipe, X_train_b, y_train_b, cv=skf, 
                                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 
                                return_train_score=True)
        pipe.fit(X_train_b, y_train_b)
        pipes_b[nama] = pipe

        y_pred_test = pipe.predict(X_test_b)

        res_skenario_b.append({
            "Model": nama,
            "CV Train F1 (%)": round(cv_res['train_f1_macro'].mean() * 100, 2),
            "CV Val F1 (%)": round(cv_res['test_f1_macro'].mean() * 100, 2),
            "CV Std (%)": round(cv_res['test_f1_macro'].std() * 100, 2),
            "Test Acc (%)": round(accuracy_score(y_test_b, y_pred_test) * 100, 2),
            "Test Prec (%)": round(precision_score(y_test_b, y_pred_test, average='macro', zero_division=0) * 100, 2),
            "Test Rec (%)": round(recall_score(y_test_b, y_pred_test, average='macro', zero_division=0) * 100, 2),
            "Test F1 (%)": round(f1_score(y_test_b, y_pred_test, average='macro', zero_division=0) * 100, 2)
        })

    df_res_b = pd.DataFrame(res_skenario_b)
    print(df_res_b.to_string(index=False))

    # Pemilihan Model Terbaik
    best_model_name = df_res_b.sort_values("CV Val F1 (%)", ascending=False).iloc[0]["Model"]
    best_pipe = pipes_b[best_model_name]
    print(f"\nModel Terbaik Berdasarkan 5-Fold Stratified CV: {best_model_name}")

    # Simpan Model Terpilih ke File Pickle (.pkl)
    with open("model_klasifikasi_likuiditas.pkl", "wb") as f:
        pickle.dump(best_pipe, f)

    # ============================================================
    # GENERASI 6 GRAFIK UTAMA EVALUASI MATERI (%)
    # ============================================================
    print("\n--- Membuat & Menyimpan 6 Grafik Visualisasi Evaluasi Akademis (%) ---")
    x = np.arange(len(df_res_b))

    # Grafik 1: Perbandingan Metrik Evaluasi Model (P13 Slide 12-15)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.18
    ax.bar(x - w*1.5, df_res_b["Test Acc (%)"], w, label="Accuracy", color="#2563EB")
    ax.bar(x - w*0.5, df_res_b["Test Prec (%)"], w, label="Precision", color="#D97706")
    ax.bar(x + w*0.5, df_res_b["Test Rec (%)"], w, label="Recall", color="#DC2626")
    ax.bar(x + w*1.5, df_res_b["Test F1 (%)"], w, label="F1-Score", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res_b["Model"], rotation=15, ha='right')
    ax.set_ylim(60, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("1. Perbandingan Metrik Evaluasi Model Klasifikasi (%)", fontsize=11, pad=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_metrik_klasifikasi.png"), dpi=300)
    plt.close()

    # Grafik 2: Confusion Matrix Model Terpilih (P12 Slide 18-21)
    y_pred_best = best_pipe.predict(X_test_b)
    cm = confusion_matrix(y_test_b, y_pred_best, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_ORDER); ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Prediksi Model"); ax.set_ylabel("Kelas Aktual")
    ax.set_title(f"2. Confusion Matrix - {best_model_name} (Data Test Murni)", fontsize=11, pad=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_confusion_matrix.png"), dpi=300)
    plt.close()

    # Grafik 3: Stabilitas 5-Fold Cross Validation (P13 Slide 23-26)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(df_res_b["Model"], df_res_b["CV Val F1 (%)"], yerr=df_res_b["CV Std (%)"], 
                fmt='o', color='#0284C7', ecolor='#EF4444', elinewidth=2, capsize=5)
    ax.set_ylim(60, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Mean CV F1-Score (%)")
    ax.set_title("3. Stabilitas Model via 5-Fold Stratified Cross-Validation (%)", fontsize=11, pad=10)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_stabilitas_cross_validation.png"), dpi=300)
    plt.close()

    # Grafik 4: Feature Importance / Explainability (P13 Slide 31-34)
    clf_final = best_pipe.named_steps['clf']
    if hasattr(clf_final, "feature_importances_"):
        imp = pd.DataFrame({
            "Fitur": best_pipe.named_steps['prep'].get_feature_names_out(),
            "Importance (%)": clf_final.feature_importances_ * 100
        }).sort_values("Importance (%)", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp["Fitur"][::-1], imp["Importance (%)"][::-1], color="#0D9488")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title("4. Feature Importance (Kontribusi Fitur dalam %)", fontsize=11, pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance.png"), dpi=300)
        plt.close()

    # Grafik 5: Ablation Study - Skenario A vs B (P13 Slide 38-40)
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.35
    ax.bar(x - w/2, df_res_a["CV Val F1 (%)"], w, label="Skenario A (Tanpa Deviasi Harga)", color="#94A3B8")
    ax.bar(x + w/2, df_res_b["CV Val F1 (%)"], w, label="Skenario B (Dengan Deviasi Harga)", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res_b["Model"], rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("5. Ablation Study: Dampak Fitur Deviasi Harga terhadap CV F1-Score (%)", fontsize=11, pad=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_ablation_study_comparison.png"), dpi=300)
    plt.close()

    # Grafik 6: Check Overfitting - Train F1 vs Validation F1 (P12 Slide 28-30)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df_res_b["Model"], df_res_b["CV Train F1 (%)"], marker='o', label="Train F1 (%)", color="#2563EB", linewidth=2)
    ax.plot(df_res_b["Model"], df_res_b["CV Val F1 (%)"], marker='s', label="Validation F1 (%)", color="#DC2626", linewidth=2, linestyle='--')
    ax.set_ylim(60, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("6. Evaluasi Overfitting: Train vs Validation F1-Score (%)", fontsize=11, pad=10)
    plt.xticks(rotation=15, ha='right')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_overfitting_check.png"), dpi=300)
    plt.close()

    # 4. Menyimpan Ringkasan Metadata Evaluasi JSON
    meta_eval = {
        "skenario_a_ablation_study": df_res_a.to_dict(orient="records"),
        "skenario_b_model_utama": df_res_b.to_dict(orient="records"),
        "model_terpilih": best_model_name,
        "tanggal_latih": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("ringkasan_evaluasi_klasifikasi.json", "w") as f:
        json.dump(meta_eval, f, indent=2)

    print(f"\n[SUKSES] Step 4 Selesai! Model & 6 Gambar Grafik Evaluasi diselaraskan di '{OUTPUT_DIR}/'.")
    print("=" * 85)


if __name__ == "__main__":
    main()