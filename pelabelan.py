"""
pelabelan.py - Step 2: Model Harga Wajar (P5) & Pembentukan Label Proxy Murni (P12)

Kriteria Bebas Kebocoran Data (Data Leakage Prevention):
1. Train/Test Split (80:20) dilakukan SEBELUM menghitung harga wajar dan ambang tertile.
2. Model Harga Wajar menggunakan Log-Target Transformation + Gradient Boosting Regressor.
3. Out-of-Fold (5-Fold CV) HANYA dihitung pada Data Training.
4. Data Test memprediksi harga wajar murni menggunakan model regresi yang dilatih dari Data Training.
5. Ambang Tertile (Q33 & Q67) dihitung HANYA dari deviasi_persen Data Training, lalu diterapkan ke Data Test.

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             mean_absolute_percentage_error, r2_score)

try:
    from processing import AMBANG_KEYAKINAN_SAMPEL, pilih_harga_wajar, klip_deviasi, DEVIASI_MIN, DEVIASI_MAX
    hitung_harga_wajar_fallback = pilih_harga_wajar
except ImportError:
    AMBANG_KEYAKINAN_SAMPEL = 15
    DEVIASI_MIN = -50.0
    DEVIASI_MAX = 100.0
    def klip_deviasi(val):
        return float(np.clip(val, DEVIASI_MIN, DEVIASI_MAX))

warnings.filterwarnings("ignore")

INPUT_FILE = "dataset_olx_bersih.csv"
OUTPUT_DATASET = "dataset_olx_berlabel.csv"
MODEL_FILE = "model_regresi_harga.pkl"
META_FILE = "pelabelan_meta.json"

def main():
    print("=" * 80)
    print("   STEP 2: MODEL HARGA WAJAR (P5) & PELABELAN PROXY BEBAS DATA LEAKAGE (P12)   ")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File '{INPUT_FILE}' tidak ditemukan. Jalankan processing.py terlebih dahulu.")

    df = pd.read_csv(INPUT_FILE)
    print(f"Total dataset bersih dimuat : {len(df):,} baris")

    fitur_num = ['usia_mobil', 'jarak_tempuh', 'km_per_tahun', 'indikasi_kredit_tdp']
    fitur_cat = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']

    X = df[fitur_num + fitur_cat]
    y = df['harga']

    # SPLIT DATA TRAIN & TEST (80:20) MURNI (Tanpa stratify agar aman dari merek satuan)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42
    )

    df['is_train'] = 0
    df.loc[train_idx, 'is_train'] = 1

    print(f"Jumlah Data Training (80%) : {len(train_idx):,} baris")
    print(f"Jumlah Data Testing  (20%) : {len(test_idx):,} baris (Diisolasi)")

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), fitur_cat),
        ('num', StandardScaler(), fitur_num)
    ])

    regressor_gb = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1
    )

    model_pipeline = Pipeline([
        ('prep', preprocessor),
        ('reg', regressor_gb)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_train_oof = cross_val_predict(model_pipeline, X_train, y_train, cv=kf)

    mae_cv = mean_absolute_error(y_train, y_train_oof)
    rmse_cv = np.sqrt(mean_squared_error(y_train, y_train_oof))
    mape_cv = mean_absolute_percentage_error(y_train, y_train_oof) * 100
    r2_cv = r2_score(y_train, y_train_oof)

    print("\n--- Performa Model Regresi Harga Wajar (5-Fold CV pada Data Train) ---")
    print(f"MAE  : Rp {mae_cv:,.0f}")
    print(f"RMSE : Rp {rmse_cv:,.0f}")
    print(f"MAPE : {mape_cv:.2f}%")
    print(f"R^2  : {r2_cv:.4f}")

    model_pipeline.fit(X_train, y_train)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_pipeline, f)

    df.loc[train_idx, 'harga_wajar'] = y_train_oof
    df.loc[test_idx, 'harga_wajar'] = model_pipeline.predict(X_test)

    df['deviasi_persen'] = ((df['harga'] - df['harga_wajar']) / df['harga_wajar']) * 100.0
    df['deviasi_persen'] = df['deviasi_persen'].apply(klip_deviasi)

    deviasi_train = df.loc[train_idx, 'deviasi_persen']
    q33 = float(np.percentile(deviasi_train, 33.333))
    q67 = float(np.percentile(deviasi_train, 66.667))

    print("\n--- Ambang Tertile Likuiditas (Dihitung HANYA dari Data Train) ---")
    print(f"Q33 (Batas Cepat Laku)  : {q33:+.2f}%")
    print(f"Q67 (Batas Lambat Laku) : {q67:+.2f}%")

    def beri_label(dev):
        if dev <= q33:
            return "Cepat"
        elif dev <= q67:
            return "Sedang"
        else:
            return "Lambat"

    df['status_likuiditas'] = df['deviasi_persen'].apply(beri_label)

    mask_turun = (df['status_likuiditas'] == 'Cepat') & (df['km_per_tahun'] > 30000) & (df['ada_garansi'] == 'Tidak')
    df.loc[mask_turun, 'status_likuiditas'] = 'Sedang'

    print("\n--- Distribusi Label Likuiditas Hasil Pelabelan ---")
    print(df['status_likuiditas'].value_counts(normalize=True) * 100)

    df.to_csv(OUTPUT_DATASET, index=False)

    meta_data = {
        "q33_tertile": round(q33, 4),
        "q67_tertile": round(q67, 4),
        "regresi_eval": {
            "mae_cv": round(mae_cv, 2),
            "rmse_cv": round(rmse_cv, 2),
            "mape_cv": round(mape_cv, 2),
            "r2_cv": round(r2_cv, 4)
        },
        "distribusi_label": df['status_likuiditas'].value_counts().to_dict(),
        "total_train": int(len(train_idx)),
        "total_test": int(len(test_idx))
    }

    with open(META_FILE, "w") as f:
        json.dump(meta_data, f, indent=2)

    print(f"\n[SUKSES] Step 2 Selesai! File '{OUTPUT_DATASET}' & '{MODEL_FILE}' berhasil diperbarui.")
    print("=" * 80)

if __name__ == "__main__":
    main()