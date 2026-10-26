"""
pelabelan.py - Step 2: Model Harga Wajar (P5) & Pembentukan Label Proxy Murni (P12)

Kriteria Bebas Kebocoran Data (Data Leakage Prevention):
1. Train/Test Split (80:20) dilakukan SEBELUM menghitung harga wajar dan ambang tertile.
2. Model Harga Wajar membandingkan Baseline (Linear Regression, Ridge) dengan Gradient Boosting Regressor (Target Log-Price).
3. Evaluasi Regresi: MAE, MSE, RMSE, MAPE, dan R^2 + Analisis Residual.
4. Out-of-fold (5-Fold CV) HANYA dihitung pada Data Training.
5. Data Test memprediksi harga wajar murni menggunakan model regresi yang dilatih dari Data Training.
6. Ambang Tertile (Q33 & Q67) dihitung HANYA dari deviasi_persen Data Training, lalu diterapkan ke Data Test.

Output:
- dataset_olx_berlabel.csv
- model_regresi_harga.pkl
- pelabelan_meta.json
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             mean_absolute_percentage_error, r2_score, root_mean_squared_error)

from processing import AMBANG_KEYAKINAN_SAMPEL, hitung_harga_wajar_fallback, klip_deviasi, DEVIASI_MIN, DEVIASI_MAX

warnings.filterwarnings("ignore")

INPUT_FILE = "dataset_olx_bersih.csv"
OUTPUT_DATASET = "dataset_olx_berlabel.csv"
MODEL_FILE = "model_regresi_harga.pkl"
META_FILE = "pelabelan_meta.json"

categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
numerical_cols = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']
fitur_harga = categorical_cols + numerical_cols

def buat_prep():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

def eval_regresi(y_true, y_pred, nama_model):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": nama_model,
        "MAE (Rp)": round(mae, 0),
        "MSE": round(mse, 0),
        "RMSE (Rp)": round(rmse, 0),
        "MAPE (%)": round(mape, 2),
        "R2 Score": round(r2, 4)
    }

def main():
    print("=" * 80)
    print("   STEP 2: MODEL HARGA WAJAR (P5) & PROXY LABELING BEBAS LEAKAGE (P12)   ")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File '{INPUT_FILE}' tidak ditemukan. Jalankan processing.py terlebih dahulu.")

    df = pd.read_csv(INPUT_FILE)
    df_model = df.dropna(subset=fitur_harga + ['harga']).copy().reset_index(drop=True)
    print(f"Total Sampel Dataset Bersih: {len(df_model):,} baris")

    # ============================================================
    # 1. TRAIN / TEST SPLIT TERLEBIH DAHULU (P12: MENCEGAH LEAKAGE)
    # ============================================================
    train_idx, test_idx = train_test_split(df_model.index, test_size=0.2, random_state=42)
    df_model['is_train'] = 0
    df_model.loc[train_idx, 'is_train'] = 1

    df_train = df_model.loc[train_idx].copy()
    df_test = df_model.loc[test_idx].copy()

    print(f"Data Training : {len(df_train):,} baris (80%)")
    print(f"Data Testing  : {len(df_test):,} baris (20%) [Diisolasi sampai evaluasi]")

    # ============================================================
    # 2. PERBANDINGAN MODEL REGRESI (Materi P5 & P12)
    # ============================================================
    print("\n--- Membandingkan Baseline & Model Utama Regresi (Log-Target) ---")
    kandidat_regresi = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Gradient Boosting Regressor (Utama)": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    }

    hasil_regresi_cv = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for nama, reg in kandidat_regresi.items():
        pipe_reg = Pipeline([
            ('prep', buat_prep()),
            ('reg', TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1))
        ])
        
        # Out-of-fold prediction HANYA pada data train
        oof_pred = cross_val_predict(pipe_reg, df_train[fitur_harga], df_train['harga'], cv=kf, n_jobs=-1)
        res = eval_regresi(df_train['harga'], oof_pred, nama)
        hasil_regresi_cv.append(res)

    df_eval_reg = pd.DataFrame(hasil_regresi_cv)
    print(df_eval_reg.to_string(index=False))

    # Latih Model Utama (Gradient Boosting) pada SELURUH Data Train
    model_harga_terpilih = Pipeline([
        ('prep', buat_prep()),
        ('reg', TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
            func=np.log1p, inverse_func=np.expm1
        ))
    ])
    model_harga_terpilih.fit(df_train[fitur_harga], df_train['harga'])

    # ============================================================
    # 3. ESTIMASI HARGA WAJAR & FALLBACK
    # ============================================================
    # A. Data Train: Menggunakan Out-Of-Fold + Fallback Leave-One-Out Train
    oof_train = cross_val_predict(model_harga_terpilih, df_train[fitur_harga], df_train['harga'], cv=kf, n_jobs=-1)
    df_train['harga_wajar_model'] = np.clip(oof_train, 1_000_000, None)

    sampel_lain_train = df_train['jumlah_sampel_kategori'] - 1
    hw_train, sumber_train = [], []
    for idx, r in df_train.iterrows():
        hw = float(r['harga_wajar_model'])
        sm = "model ML"
        if sampel_lain_train[idx] < AMBANG_KEYAKINAN_SAMPEL:
            hf, s = hitung_harga_wajar_fallback(df_train, r['merek'], r['model'], r['usia_mobil'], exclude_idx=idx)
            if hf is not None:
                hw, sm = hf, s
        hw_train.append(hw)
        sumber_train.append(sm)

    df_train['harga_wajar_final'] = hw_train
    df_train['sumber_harga_wajar'] = sumber_train

    # B. Data Test: Murni Memakai Model Train (TIDAK OOF) + Fallback dari Reference Train
    test_pred = model_harga_terpilih.predict(df_test[fitur_harga])
    df_test['harga_wajar_model'] = np.clip(test_pred, 1_000_000, None)

    hw_test, sumber_test = [], []
    for idx, r in df_test.iterrows():
        hw = float(r['harga_wajar_model'])
        sm = "model ML"
        if r['jumlah_sampel_kategori'] < AMBANG_KEYAKINAN_SAMPEL:
            hf, s = hitung_harga_wajar_fallback(df_train, r['merek'], r['model'], r['usia_mobil'], exclude_idx=None)
            if hf is not None:
                hw, sm = hf, s
        hw_test.append(hw)
        sumber_test.append(sm)

    df_test['harga_wajar_final'] = hw_test
    df_test['sumber_harga_wajar'] = sumber_test

    # ============================================================
    # 4. KALKULASI DEVIASI HARGA & TERTILE AMBANG (HANYA DARI TRAIN)
    # ============================================================
    df_train['deviasi_persen'] = ((df_train['harga'] - df_train['harga_wajar_final']) / df_train['harga_wajar_final'] * 100).clip(DEVIASI_MIN, DEVIASI_MAX)
    df_test['deviasi_persen'] = ((df_test['harga'] - df_test['harga_wajar_final']) / df_test['harga_wajar_final'] * 100).clip(DEVIASI_MIN, DEVIASI_MAX)

    # Hitung Tertile HANYA dari Data Training
    q33 = float(df_train['deviasi_persen'].quantile(0.33))
    q67 = float(df_train['deviasi_persen'].quantile(0.67))

    print(f"\nAmbang Tertile (Dihitung HANYA dari Data Train):")
    print(f"  - Q33 (Batas Cepat Laku) : {q33:.2f}%")
    print(f"  - Q67 (Batas Lambat Laku): {q67:.2f}%")

    # Terapkan Labeling ke Train dan Test menggunakan Ambang Train
    for df_sub in [df_train, df_test]:
        df_sub['status_likuiditas'] = np.where(df_sub['deviasi_persen'] <= q33, "Cepat",
                                     np.where(df_sub['deviasi_persen'] <= q67, "Sedang", "Lambat"))
        turun = ((df_sub['status_likuiditas'] == "Cepat") & (df_sub['km_per_tahun'] > 30000) & (df_sub['ada_garansi'] == 'Tidak'))
        df_sub.loc[turun, 'status_likuiditas'] = "Sedang"

    # Gabungkan Kembali Dataset
    df_berlabel = pd.concat([df_train, df_test]).sort_index()

    # Evaluasi Kinerja Regresi pada Data Test
    eval_test_reg = eval_regresi(df_test['harga'], df_test['harga_wajar_final'], "Gradient Boosting (Test Set)")
    print(f"\nKinerja Harga Wajar pada Data Test Murni:")
    print(f"  - MAPE : {eval_test_reg['MAPE (%)']}%")
    print(f"  - MAE  : Rp {eval_test_reg['MAE (Rp)']:,.0f}")
    print(f"  - R^2  : {eval_test_reg['R2 Score']}")

    # ============================================================
    # 5. SIMPAN MODEL & METADATA
    # ============================================================
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_harga_terpilih, f)

    df_berlabel.to_csv(OUTPUT_DATASET, index=False, encoding='utf-8-sig')

    meta_pelabelan = {
        "q33_train": round(q33, 4),
        "q67_train": round(q67, 4),
        "regresi_komparasi_train_cv": hasil_regresi_cv,
        "regresi_test_performance": eval_test_reg,
        "distribusi_label_train": df_train['status_likuiditas'].value_counts().to_dict(),
        "distribusi_label_test": df_test['status_likuiditas'].value_counts().to_dict()
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_pelabelan, f, indent=2)

    print(f"\n[SUKSES] Step 2 Selesai! Dataset berlabel tersimpan di '{OUTPUT_DATASET}'.")
    print("=" * 80)

if __name__ == "__main__":
    main()