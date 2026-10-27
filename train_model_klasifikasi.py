"""
AutoLaku - Pelatihan Model Klasifikasi Kecepatan Laku (Cepat/Sedang/Lambat)
Versi 2: Dirombak sesuai materi P6, P9, P12, P13.

Perubahan utama:
  - Membaca dataset_olx_berlabel.csv (keluaran pelabelan.py, split sudah fix).
  - Kandidat diperluas: Logistic Regression, KNN (K=3,5,7,9), Naive Bayes
    (Gaussian, Categorical), Decision Tree, Random Forest, Gradient Boosting.
  - 5-fold Stratified CV dengan tuning hyperparameter.
  - Estimasi akhir dengan nested CV.
  - Laporan tiap model: F1 ± std, train vs val (cek overfitting/underfitting),
    RMSE ordinal, biaya kesalahan berbobot.
  - Dua skenario: TANPA deviasi_persen (perbandingan adil) dan DENGAN
    (model untuk aplikasi).
  - Interpretasi: permutation importance, koefisien LR, aturan DT.
  - Kriteria pilih: F1, stabilitas, selisih train-val, monotonisitas,
    keyakinan wajar, interpretabilitas. Selisih < 1 std tidak bermakna.
  - Test set dibuka SEKALI di akhir.

Keluaran:
  - model_klasifikasi_likuiditas.pkl
  - autolaku_meta.json (diperbarui)
  - output_klasifikasi_<tanggal>/  (grafik dan ringkasan evaluasi)
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix,
                              classification_report, make_scorer,
                              mean_squared_error)
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from model_khusus import NaiveBayesDiskret

from processing import AMBANG_KEYAKINAN_SAMPEL, DEVIASI_MIN, DEVIASI_MAX

warnings.filterwarnings("ignore")
OUTPUT_DIR = f"output_klasifikasi_{datetime.now().strftime('%d-%m-%Y')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# KONSTANTA
# ============================================================
DATASET_BERLABEL = "dataset_olx_berlabel.csv"
META_FILE = "autolaku_meta.json"
MODEL_KLS_FILE = "model_klasifikasi_likuiditas.pkl"
LABEL_ORDER = ["Cepat", "Sedang", "Lambat"]
RANK = {l: i for i, l in enumerate(LABEL_ORDER)}
RANDOM_STATE = 42
CV_FOLDS = 5


def rmse_ordinal(y_true, y_pred):
    """RMSE berbasis peringkat ordinal (Cepat=0, Sedang=1, Lambat=2)."""
    r_true = np.array([RANK.get(y, 1) for y in y_true])
    r_pred = np.array([RANK.get(y, 1) for y in y_pred])
    return float(np.sqrt(mean_squared_error(r_true, r_pred)))


def biaya_berbobot(y_true, y_pred):
    """Biaya kesalahan berbobot selisih kelas. Meleset 2 kelas = biaya 2x."""
    r_true = np.array([RANK.get(y, 1) for y in y_true])
    r_pred = np.array([RANK.get(y, 1) for y in y_pred])
    return float(np.abs(r_true - r_pred).mean())


def pelanggaran_monoton(pipe, X_ref, fitur_kls, n=300):
    """Persentase unit yang vonisnya menjadi LEBIH CEPAT saat harga dinaikkan."""
    if 'deviasi_persen' not in fitur_kls:
        return np.nan
    sampel = X_ref.sample(min(n, len(X_ref)), random_state=1)
    grid = np.linspace(-80, 250, 34)
    besar = sampel.loc[sampel.index.repeat(len(grid))].copy()
    besar['deviasi_persen'] = np.tile(grid, len(sampel))
    rank = np.array([RANK.get(p, 1) for p in pipe.predict(besar[fitur_kls])])
    rank = rank.reshape(len(sampel), len(grid))
    return float((np.diff(rank, axis=1) < 0).any(axis=1).mean() * 100)


def rerata_keyakinan_puncak(pipe, X_ref, fitur_kls):
    """Rata-rata probabilitas kelas yang dipilih model."""
    clf_step = pipe.named_steps.get('clf')
    if clf_step is None or not hasattr(clf_step, "predict_proba"):
        return np.nan
    try:
        proba = pipe.predict_proba(X_ref[fitur_kls])
        return float(proba.max(axis=1).mean())
    except Exception:
        return np.nan


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 75)
    print("PELATIHAN MODEL KLASIFIKASI (P6, P9, P12, P13)")
    print("=" * 75)

    if not os.path.exists(DATASET_BERLABEL):
        raise FileNotFoundError(
            f"'{DATASET_BERLABEL}' tidak ditemukan. Jalankan pelabelan.py dulu.")

    # Muat metadata dari pelabelan
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            meta = json.load(f)
    else:
        meta = {}

    df_all = pd.read_csv(DATASET_BERLABEL)
    df_train = df_all[df_all['split'] == 'train'].copy()
    df_test = df_all[df_all['split'] == 'test'].copy()
    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

    # Tentukan kolom fitur
    categorical_cols = meta.get('categorical_cols',
        ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi',
         'riwayat_negatif', 'catatan_servis', 'pajak_hidup'])
    numerical_cols = meta.get('numerical_cols',
        ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun'])

    # Filter hanya kolom yang tersedia
    categorical_cols = [c for c in categorical_cols if c in df_train.columns]
    numerical_cols = [c for c in numerical_cols if c in df_train.columns]

    # Fitur final dari analisis_fitur.py (konstan / redundan dibuang)
    dibuang_final = {}
    if os.path.exists("fitur_final.json"):
        with open("fitur_final.json", encoding="utf-8") as f_ff:
            fitur_final = json.load(f_ff)
        dipertahankan = set(fitur_final["dipertahankan"])
        dibuang_final = fitur_final.get("dibuang", {})
        categorical_cols = [c for c in categorical_cols if c in dipertahankan]
        numerical_cols = [c for c in numerical_cols if c in dipertahankan]
        print(f"Fitur mengikuti fitur_final.json. Dibuang: {dibuang_final}")
    else:
        print("[PERINGATAN] fitur_final.json tidak ada; semua fitur kandidat dipakai. Jalankan analisis_fitur.py.")

    fitur_harga = categorical_cols + numerical_cols
    fitur_kls_tanpa_deviasi = fitur_harga
    fitur_kls_dengan_deviasi = fitur_harga + ['deviasi_persen']

    y_train = df_train['status_likuiditas']
    y_test = df_test['status_likuiditas']

    def buat_prep(num_cols):
        return ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), num_cols),
        ])

    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ============================================================
    # SKENARIO 1: TANPA deviasi_persen (perbandingan adil)
    # ============================================================
    print("\n" + "=" * 75)
    print("SKENARIO 1: TANPA deviasi_persen (perbandingan adil antar algoritma)")
    print("=" * 75)

    kandidat_skenario1 = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE),
        "KNN (K=3)": KNeighborsClassifier(n_neighbors=3),
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
        "KNN (K=7)": KNeighborsClassifier(n_neighbors=7),
        "KNN (K=9)": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes (Bernoulli, diskretisasi)": NaiveBayesDiskret(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE),
    }

    def evaluasi_kandidat(kandidat, fitur_kls, nama_skenario):
        """Evaluasi semua kandidat dengan Stratified CV, return DataFrame hasil."""
        num_cols_active = [c for c in numerical_cols if c in fitur_kls]
        if 'deviasi_persen' in fitur_kls:
            num_cols_active = num_cols_active + ['deviasi_persen']

        prep = buat_prep(num_cols_active)
        hasil = []

        print(f"\n{'Model':25s} | {'F1_val':>8s} | {'+/-Std':>6s} | {'F1_trn':>8s} | "
              f"{'Gap':>7s} | {'RMSE_o':>6s}")
        print("-" * 80)

        for nama, clf in kandidat.items():
            pipe = Pipeline([('prep', prep), ('clf', clf)])

            # Cross-validation dengan skor training dan validasi
            cv_results = cross_validate(
                pipe, df_train[fitur_kls], y_train,
                cv=skf, scoring='f1_macro',
                return_train_score=True, n_jobs=-1)

            f1_val_mean = cv_results['test_score'].mean()
            f1_val_std = cv_results['test_score'].std()
            f1_train_mean = cv_results['train_score'].mean()
            gap = f1_train_mean - f1_val_mean

            # OOF prediction untuk RMSE ordinal dan biaya berbobot
            oof_pred = cross_val_predict(
                pipe, df_train[fitur_kls], y_train, cv=skf, n_jobs=-1)
            rmse_o = rmse_ordinal(y_train, oof_pred)
            biaya = biaya_berbobot(y_train, oof_pred)

            # Accuracy OOF
            acc_oof = accuracy_score(y_train, oof_pred)

            hasil.append({
                "Model": nama,
                "F1_Val_Mean": f1_val_mean,
                "F1_Val_Std": f1_val_std,
                "F1_Train_Mean": f1_train_mean,
                "Gap_Train_Val": gap,
                "Accuracy_OOF": acc_oof,
                "RMSE_Ordinal": rmse_o,
                "Biaya_Berbobot": biaya,
            })

            print(f"  {nama:23s} | {f1_val_mean:>7.4f} | {f1_val_std:>5.4f} | "
                  f"{f1_train_mean:>7.4f} | {gap:>+6.4f} | {rmse_o:>5.4f}")

        df_hasil = pd.DataFrame(hasil)

        # Tambahkan catatan overfitting/underfitting
        for _, row in df_hasil.iterrows():
            gap = row['Gap_Train_Val']
            if gap > 0.10:
                status = "OVERFITTING (gap>0.10)"
            elif gap < -0.02:
                status = "UNDERFITTING"
            else:
                status = "OK"
            print(f"    {row['Model']:25s} -> {status}")

        return df_hasil

    hasil_s1 = evaluasi_kandidat(kandidat_skenario1, fitur_kls_tanpa_deviasi, "Tanpa Deviasi")

    # ============================================================
    # SKENARIO 2: DENGAN deviasi_persen (model untuk aplikasi)
    # ============================================================
    print("\n" + "=" * 75)
    print("SKENARIO 2: DENGAN deviasi_persen (model untuk aplikasi)")
    print("=" * 75)

    kandidat_skenario2 = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE),
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Bernoulli, diskretisasi)": NaiveBayesDiskret(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE),
    }

    hasil_s2 = evaluasi_kandidat(kandidat_skenario2, fitur_kls_dengan_deviasi, "Dengan Deviasi")

    # ============================================================
    # PEMILIHAN MODEL (Skenario 2 — model untuk aplikasi)
    # ============================================================
    print("\n" + "=" * 75)
    print("PEMILIHAN MODEL (Skenario 2 — untuk aplikasi)")
    print("=" * 75)

    # Latih ulang semua model skenario 2 pada SELURUH training untuk monoton & keyakinan
    num_cols_s2 = [c for c in numerical_cols if c in fitur_kls_dengan_deviasi]
    if 'deviasi_persen' in fitur_kls_dengan_deviasi:
        num_cols_s2 = num_cols_s2 + ['deviasi_persen']
    prep_s2 = buat_prep(num_cols_s2)

    pipes_s2 = {}
    for nama, clf in kandidat_skenario2.items():
        p = Pipeline([('prep', prep_s2), ('clf', clf)])
        p.fit(df_train[fitur_kls_dengan_deviasi], y_train)
        pipes_s2[nama] = p

    # Tambahkan metrik monotonisitas dan keyakinan ke hasil_s2
    for i, (nama, pipe) in enumerate(pipes_s2.items()):
        langgar = pelanggaran_monoton(pipe, df_train, fitur_kls_dengan_deviasi)
        keyakinan = rerata_keyakinan_puncak(pipe, df_train, fitur_kls_dengan_deviasi)
        hasil_s2.loc[hasil_s2['Model'] == nama, 'Pelanggaran_Monoton_%'] = langgar
        hasil_s2.loc[hasil_s2['Model'] == nama, 'Rerata_Keyakinan_Puncak'] = keyakinan

    print("\nMetrik pemilihan:")
    print(hasil_s2[['Model', 'F1_Val_Mean', 'F1_Val_Std', 'Gap_Train_Val',
                     'Pelanggaran_Monoton_%', 'Rerata_Keyakinan_Puncak']].to_string(index=False))

    # Kriteria pemilihan:
    # 1. HARUS monoton (0% pelanggaran) — syarat mutlak
    # 2. Tidak overconfident (keyakinan < 0.97)
    # 3. F1-macro tertinggi
    # 4. Selisih < 1 std tidak dianggap bermakna
    monoton = hasil_s2[hasil_s2['Pelanggaran_Monoton_%'] == 0]
    if monoton.empty:
        monoton = hasil_s2.sort_values('Pelanggaran_Monoton_%').head(1)
        print("\n[PERINGATAN] Tidak ada model 100% monoton; dipilih yang paling mendekati.")

    wajar = monoton[monoton['Rerata_Keyakinan_Puncak'] <= 0.97]
    pool = wajar if len(wajar) else monoton
    if wajar.empty and len(monoton) > 1:
        print("[INFO] Semua model monoton overconfident; dipilih F1-macro terbaik.")

    best_row = pool.sort_values('F1_Val_Mean', ascending=False).iloc[0]
    best_name = best_row['Model']
    best_pipe = pipes_s2[best_name]
    print(f"\n  Model terpilih: {best_name}")
    print(f"  F1-macro (CV): {best_row['F1_Val_Mean']:.4f} +/- {best_row['F1_Val_Std']:.4f}")
    print(f"  Gap train-val: {best_row['Gap_Train_Val']:+.4f}")
    print(f"  Monotonisitas: {best_row.get('Pelanggaran_Monoton_%', 'N/A')}")

    # ============================================================
    # TUNING HYPERPARAMETER + NESTED CV (P13 slide 31-33)
    # ============================================================
    print("\n" + "=" * 75)
    print("TUNING HYPERPARAMETER (RandomizedSearchCV) + NESTED CV")
    print("=" * 75)
    GRID = {
        "Random Forest": {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [8, 12, 16, None],
                          'clf__min_samples_leaf': [1, 2, 4], 'clf__max_features': ['sqrt', 0.5]},
        "Gradient Boosting": {'clf__n_estimators': [100, 150, 250], 'clf__learning_rate': [0.05, 0.1],
                              'clf__max_depth': [2, 3, 4], 'clf__subsample': [0.8, 1.0]},
        "Decision Tree": {'clf__max_depth': [4, 6, 8, 10, None], 'clf__min_samples_leaf': [1, 3, 5, 10]},
        "Logistic Regression": {'clf__C': [0.01, 0.1, 1, 10, 100]},
        "KNN (K=5)": {'clf__n_neighbors': [3, 5, 7, 9, 11]},
        "Naive Bayes (Bernoulli, diskretisasi)": {'clf__alpha': [0.1, 0.5, 1.0, 2.0], 'clf__n_bins': [4, 6, 8, 10]},
    }
    N_ITER_TUNING = 8
    info_tuning = {"dilakukan": False}
    if best_name in GRID:
        X_trn = df_train[fitur_kls_dengan_deviasi]
        pipa = Pipeline([('prep', clone(prep_s2)), ('clf', clone(kandidat_skenario2[best_name]))])
        inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
        cari = RandomizedSearchCV(pipa, GRID[best_name], n_iter=N_ITER_TUNING, cv=inner,
                                  scoring='f1_macro', random_state=RANDOM_STATE, n_jobs=-1)
        skor_nested = cross_val_score(cari, X_trn, y_train, cv=skf, scoring='f1_macro')
        cari.fit(X_trn, y_train)
        f1_def, std_def = float(best_row['F1_Val_Mean']), float(best_row['F1_Val_Std'])
        langgar_tuned = pelanggaran_monoton(cari.best_estimator_, df_train, fitur_kls_dengan_deviasi)
        keyakinan_tuned = rerata_keyakinan_puncak(cari.best_estimator_, df_train, fitur_kls_dengan_deviasi)
        lebih_baik = skor_nested.mean() > f1_def + std_def     # selisih < 1 std dianggap tidak bermakna
        aman = (langgar_tuned == 0) and (keyakinan_tuned <= 0.97)
        pakai = bool(lebih_baik and aman)
        print(f"  Model            : {best_name}")
        print(f"  F1 default (CV)  : {f1_def:.4f} +/- {std_def:.4f}")
        print(f"  F1 nested CV     : {skor_nested.mean():.4f} +/- {skor_nested.std():.4f}  (estimasi tanpa bias utk prosedur tuning)")
        print(f"  Parameter terbaik: {cari.best_params_}")
        print(f"  Monoton/keyakinan: {langgar_tuned}% pelanggaran | keyakinan {keyakinan_tuned:.3f}")
        alasan = ("lebih baik > 1 std dan aman" if pakai else
                  "selisih dengan default tidak bermakna (< 1 std)" if not lebih_baik else
                  "ditolak: melanggar monotonisitas atau overconfident")
        print(f"  Keputusan        : {'PAKAI model hasil tuning' if pakai else 'PAKAI parameter default'} ({alasan})")
        if pakai:
            best_pipe = cari.best_estimator_
        info_tuning = {"dilakukan": True, "metode": f"RandomizedSearchCV n_iter={N_ITER_TUNING}, inner CV=3, outer CV={CV_FOLDS}",
                       "parameter_terbaik": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                                             for k, v in cari.best_params_.items()},
                       "f1_nested_mean": round(float(skor_nested.mean()), 4),
                       "f1_nested_std": round(float(skor_nested.std()), 4),
                       "f1_default_cv": round(f1_def, 4), "dipakai": pakai, "alasan": alasan}
    else:
        print(f"  Tidak ada grid untuk {best_name}; tuning dilewati.")
    keyakinan_final = rerata_keyakinan_puncak(best_pipe, df_train, fitur_kls_dengan_deviasi)

    # ============================================================
    # EVALUASI PADA TEST SET (dibuka SEKALI)
    # ============================================================
    print("\n" + "=" * 75)
    print("EVALUASI PADA TEST SET (dibuka sekali)")
    print("=" * 75)

    y_pred_test = best_pipe.predict(df_test[fitur_kls_dengan_deviasi])
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    prec_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    rmse_o_test = rmse_ordinal(y_test, y_pred_test)
    biaya_test = biaya_berbobot(y_test, y_pred_test)

    print(f"\n  Accuracy  : {acc_test:.4f}")
    print(f"  F1-Macro  : {f1_test:.4f}")
    print(f"  Precision : {prec_test:.4f}")
    print(f"  Recall    : {rec_test:.4f}")
    print(f"  RMSE Ord. : {rmse_o_test:.4f}")
    print(f"  Biaya Avg : {biaya_test:.4f}")
    print(f"\n{classification_report(y_test, y_pred_test, labels=LABEL_ORDER)}")

    # ============================================================
    # UJI ABLATION: Skenario 1 vs Skenario 2 (model terpilih)
    # ============================================================
    print("\n" + "=" * 75)
    print(f"UJI ABLATION ({best_name}): TANPA vs DENGAN deviasi_persen")
    print("=" * 75)

    # Latih model yang sama tanpa deviasi
    num_cols_s1 = numerical_cols.copy()
    prep_s1 = buat_prep(num_cols_s1)
    clf_ablation = clone(kandidat_skenario2[best_name])
    pipe_ablation = Pipeline([('prep', prep_s1), ('clf', clf_ablation)])
    pipe_ablation.fit(df_train[fitur_kls_tanpa_deviasi], y_train)
    yp_abl = pipe_ablation.predict(df_test[fitur_kls_tanpa_deviasi])
    acc_abl = accuracy_score(y_test, yp_abl)
    f1_abl = f1_score(y_test, yp_abl, average='macro')

    selisih_acc = acc_test - acc_abl
    selisih_f1 = f1_test - f1_abl
    print(f"  Dengan deviasi_persen   : Acc {acc_test:.4f} | F1 {f1_test:.4f}")
    print(f"  Tanpa deviasi_persen    : Acc {acc_abl:.4f} | F1 {f1_abl:.4f}")
    print(f"  Selisih Acc: {selisih_acc:+.4f} | Selisih F1: {selisih_f1:+.4f}")
    if selisih_acc > 0.15:
        print("  -> Penurunan besar: performa didominasi deviasi_persen (sirkularitas).")
    else:
        print("  -> Penurunan kecil: fitur lain juga berkontribusi nyata.")

    # ============================================================
    # INTERPRETASI MODEL
    # ============================================================
    print("\n" + "=" * 75)
    print("INTERPRETASI MODEL")
    print("=" * 75)
    print("[CATATAN] Interpretasi di bawah menunjukkan ASOSIASI, BUKAN sebab-akibat.")

    # Permutation Importance
    print("\n--- Permutation Importance (pada test set) ---")
    X_test_pi = best_pipe.named_steps['prep'].transform(df_test[fitur_kls_dengan_deviasi])
    feature_names_pi = best_pipe.named_steps['prep'].get_feature_names_out()
    perm_imp = permutation_importance(
        best_pipe.named_steps['clf'], X_test_pi, y_test,
        n_repeats=10, random_state=RANDOM_STATE, scoring='f1_macro', n_jobs=-1)
    perm_df = pd.DataFrame({
        "fitur": feature_names_pi,
        "importance_mean": perm_imp.importances_mean,
        "importance_std": perm_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(perm_df.head(15).to_string(index=False))

    # Koefisien Logistic Regression (jika tersedia)
    if best_name == "Logistic Regression":
        print("\n--- Koefisien Logistic Regression (top per kelas) ---")
        clf_lr = best_pipe.named_steps['clf']
        for i, kelas in enumerate(clf_lr.classes_):
            coef = pd.Series(clf_lr.coef_[i], index=feature_names_pi)
            top = coef.abs().nlargest(10)
            print(f"\n  Kelas '{kelas}':")
            for feat in top.index:
                print(f"    {feat:40s}: {coef[feat]:+.4f}")

    # Feature Importance dari model tree-based
    clf_final = best_pipe.named_steps['clf']
    if hasattr(clf_final, "feature_importances_"):
        print(f"\n--- Feature Importance ({best_name}) ---")
        imp = pd.DataFrame({
            "fitur": feature_names_pi,
            "importance": clf_final.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(imp.head(15).to_string(index=False))

    # Aturan Decision Tree (jika terpilih)
    if best_name == "Decision Tree":
        from sklearn.tree import export_text
        print("\n--- Aturan Decision Tree (kedalaman 3) ---")
        dt_rules = export_text(clf_final, feature_names=list(feature_names_pi), max_depth=3)
        print(dt_rules)

    # ============================================================
    # SIMPAN KELUARAN
    # ============================================================
    print("\n" + "=" * 75)
    print("SIMPAN KELUARAN")
    print("=" * 75)

    with open(MODEL_KLS_FILE, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"  Model klasifikasi: {MODEL_KLS_FILE}")

    # Update metadata
    meta.update({
        "model_klasifikasi": best_name,
        "fitur_klasifikasi": fitur_kls_dengan_deviasi,
        "fitur_dibuang_klasifikasi": dibuang_final,
        "tuning": info_tuning,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "rerata_keyakinan_puncak": round(float(keyakinan_final), 4),
        "f1_macro_cv": round(float(best_row['F1_Val_Mean']), 4),
        "f1_macro_cv_std": round(float(best_row['F1_Val_Std']), 4),
        "f1_macro_test": round(f1_test, 4),
        "accuracy_test": round(acc_test, 4),
        "rmse_ordinal_test": round(rmse_o_test, 4),
        "ablation_akurasi_dengan_deviasi": round(acc_test, 4),
        "ablation_akurasi_tanpa_deviasi": round(acc_abl, 4),
        "ablation_selisih_akurasi": round(selisih_acc, 4),
        "ablation_f1_dengan_deviasi": round(f1_test, 4),
        "ablation_f1_tanpa_deviasi": round(f1_abl, 4),
        "catatan": "Label status_likuiditas dibentuk dari deviasi_persen, yang juga dipakai sbg "
                   "fitur classifier — akurasi tinggi sebagian bersifat sirkular, lihat ablation_*.",
        "tanggal_latih": datetime.now().strftime("%Y-%m-%d"),
    })
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {META_FILE}")

    # --- Simpan ringkasan evaluasi (struktur 6 bagian P13 slide 43) ---
    ringkasan = {
        "1_pemilihan_model": {
            "skenario_1_tanpa_deviasi": hasil_s1.to_dict(orient='records'),
            "skenario_2_dengan_deviasi": hasil_s2.to_dict(orient='records'),
        },
        "2_model_terpilih": best_name,
        "3_evaluasi_cv": {
            "f1_macro_mean": round(float(best_row['F1_Val_Mean']), 4),
            "f1_macro_std": round(float(best_row['F1_Val_Std']), 4),
            "gap_train_val": round(float(best_row['Gap_Train_Val']), 4),
        },
        "4_evaluasi_test": {
            "accuracy": round(acc_test, 4),
            "f1_macro": round(f1_test, 4),
            "precision_macro": round(prec_test, 4),
            "recall_macro": round(rec_test, 4),
            "rmse_ordinal": round(rmse_o_test, 4),
            "biaya_berbobot": round(biaya_test, 4),
        },
        "5_ablation": {
            "dengan_deviasi": {"acc": round(acc_test, 4), "f1": round(f1_test, 4)},
            "tanpa_deviasi": {"acc": round(acc_abl, 4), "f1": round(f1_abl, 4)},
        },
        "6_interpretasi": "Lihat permutation importance dan feature importance di output grafik.",
    }
    with open(os.path.join(OUTPUT_DIR, "ringkasan_evaluasi.json"), "w") as f:
        json.dump(ringkasan, f, indent=2, default=str)

    # --- CSV ringkasan ---
    hasil_s1.to_csv(os.path.join(OUTPUT_DIR, "metrik_skenario1_tanpa_deviasi.csv"), index=False)
    hasil_s2.to_csv(os.path.join(OUTPUT_DIR, "metrik_skenario2_dengan_deviasi.csv"), index=False)

    # ============================================================
    # VISUALISASI
    # ============================================================
    _buat_visualisasi(hasil_s1, hasil_s2, y_test, y_pred_test, best_name, best_pipe,
                      df_test, fitur_kls_dengan_deviasi, feature_names_pi, perm_df)

    print(f"\n  Grafik & ringkasan disimpan di '{OUTPUT_DIR}/'.")
    print("\n[SELESAI] Pelatihan model klasifikasi selesai.")


def _buat_visualisasi(hasil_s1, hasil_s2, y_test, y_pred_best, best_name,
                       best_pipe, df_test, fitur_kls, feature_names, perm_df):
    """Buat semua grafik visualisasi."""

    # 1. Komparasi metrik Skenario 1 (tanpa deviasi)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(hasil_s1))
    ax.bar(x - 0.175, hasil_s1["Accuracy_OOF"], 0.35, label="Accuracy (OOF)", color="#2563EB")
    ax.bar(x + 0.175, hasil_s1["F1_Val_Mean"], 0.35, label="F1-Macro (CV)", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(hasil_s1["Model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Skenario 1: TANPA deviasi_persen (perbandingan adil)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_tanpa_deviasi.png"), dpi=200)
    plt.close(fig)

    # 2. Komparasi metrik Skenario 2 (dengan deviasi)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(hasil_s2))
    ax.bar(x - 0.175, hasil_s2["Accuracy_OOF"], 0.35, label="Accuracy (OOF)", color="#2563EB")
    ax.bar(x + 0.175, hasil_s2["F1_Val_Mean"], 0.35, label="F1-Macro (CV)", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(hasil_s2["Model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Skenario 2: DENGAN deviasi_persen (model untuk aplikasi)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_komparasi_dengan_deviasi.png"), dpi=200)
    plt.close(fig)

    # 3. Confusion Matrix (test set)
    cm = confusion_matrix(y_test, y_pred_best, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_ORDER); ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual (label proxy)")
    ax.set_title(f"Confusion Matrix — {best_name} (Test Set)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_confusion_matrix_test.png"), dpi=200)
    plt.close(fig)

    # 4. Train vs Val Gap (overfitting check)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(hasil_s2))
    ax.bar(x - 0.175, hasil_s2["F1_Train_Mean"], 0.35, label="F1 Train", color="#2563EB", alpha=0.7)
    ax.bar(x + 0.175, hasil_s2["F1_Val_Mean"], 0.35, label="F1 Validasi", color="#DC2626", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(hasil_s2["Model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Train vs Validasi F1 (cek overfitting)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_train_vs_val.png"), dpi=200)
    plt.close(fig)

    # 5. RMSE Ordinal per model
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hasil_s2["Model"], hasil_s2["RMSE_Ordinal"], color="#7C3AED")
    ax.set_ylabel("RMSE Ordinal")
    ax.set_title("RMSE Ordinal per Model (Cepat=0, Sedang=1, Lambat=2)")
    ax.set_xticklabels(hasil_s2["Model"], rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(hasil_s2["RMSE_Ordinal"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_rmse_ordinal.png"), dpi=200)
    plt.close(fig)

    # 6. Permutation Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_perm = perm_df.head(15)
    ax.barh(top_perm["fitur"][::-1], top_perm["importance_mean"][::-1], color="#0D9488")
    ax.set_xlabel("Permutation Importance (F1-Macro)")
    ax.set_title(f"Permutation Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_permutation_importance.png"), dpi=200)
    plt.close(fig)

    # 7. Feature Importance (jika tree-based)
    clf_final = best_pipe.named_steps['clf']
    if hasattr(clf_final, "feature_importances_"):
        imp = pd.DataFrame({
            "fitur": feature_names,
            "importance": clf_final.feature_importances_,
        }).sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp["fitur"][::-1], imp["importance"][::-1], color="#0D9488")
        ax.set_title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "7_feature_importance.png"), dpi=200)
        plt.close(fig)

    # 8. F1 Stability (error bar)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hasil_s2["Model"], hasil_s2["F1_Val_Mean"], yerr=hasil_s2["F1_Val_Std"],
           capsize=5, color="#F59E0B", edgecolor="black")
    ax.set_ylabel("F1-Macro (± Std)")
    ax.set_title("Stabilitas F1-Macro per Model (5-Fold Stratified CV)")
    ax.set_xticklabels(hasil_s2["Model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "8_stabilitas_f1.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()