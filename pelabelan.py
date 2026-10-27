"""
AutoLaku - Pelabelan Data: Split, Model Harga Wajar, Pembuatan Label
(materi P5 klasifikasi, P12 evaluasi & kebocoran data)

Alur:
  1. Baca dataset_olx_bersih.csv (keluaran processing.py).
  2. Split train/test SEBELUM label dibuat (menutup kebocoran data P12).
  3. Bandingkan model regresi harga wajar: Linear Regression, Ridge, dan
     GradientBoosting (target log-harga), dengan metrik MAE, MSE, RMSE,
     MAPE, R², dan analisis residual.
  4. Harga wajar data training dihitung out-of-fold (5-fold CV).
     Data test memakai model yang dilatih dari SELURUH data training.
  5. Ambang tertile (Q33/Q67) dihitung hanya dari data training,
     lalu dipakai untuk data test — menutup kebocoran ambang.
  6. Penyesuaian aturan bisnis: km tinggi + tanpa garansi → turun kelas.

Keluaran:
  - dataset_olx_berlabel.csv   (dengan kolom 'split' dan 'status_likuiditas')
  - model_regresi_harga.pkl    (model terbaik, dilatih dari seluruh data training)
  - autolaku_meta.json         (ambang tertile + metadata regresi)
  - output_pelabelan/          (grafik dan laporan)
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

from sklearn.model_selection import (train_test_split, KFold, cross_val_predict,
                                     cross_val_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              mean_absolute_percentage_error, r2_score)

from processing import (AMBANG_KEYAKINAN_SAMPEL, DEVIASI_MIN, DEVIASI_MAX,
                         hitung_harga_wajar_fallback, klip_deviasi)

warnings.filterwarnings("ignore")
OUTPUT_DIR = "output_pelabelan"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# KONSTANTA
# ============================================================
DATASET_BERSIH = "dataset_olx_bersih.csv"
DATASET_BERLABEL = "dataset_olx_berlabel.csv"
MODEL_REGRESI_FILE = "model_regresi_harga.pkl"
META_FILE = "autolaku_meta.json"

CATEGORICAL_COLS = ['merek', 'model', 'transmisi', 'ada_garansi',
                    'riwayat_negatif', 'catatan_servis', 'pajak_hidup']
NUMERICAL_COLS = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']
FITUR_HARGA = CATEGORICAL_COLS + NUMERICAL_COLS
LABEL_ORDER = ["Cepat", "Sedang", "Lambat"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def buat_prep():
    """Preprocessor: one-hot encoding kategorikal + scaling numerikal."""
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
        ('num', StandardScaler(), NUMERICAL_COLS),
    ])


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 75)
    print("PELABELAN DATA (P5, P12)")
    print("=" * 75)

    if not os.path.exists(DATASET_BERSIH):
        raise FileNotFoundError(f"'{DATASET_BERSIH}' tidak ditemukan. Jalankan processing.py dulu.")

    df = pd.read_csv(DATASET_BERSIH)
    print(f"Dataset bersih dimuat: {len(df):,} baris")
    if 'is_noise' in df.columns:   # noise (data salah) tidak ikut membentuk model & label
        n_noise = int(df['is_noise'].astype(bool).sum())
        df = df[~df['is_noise'].astype(bool)].copy()
        print(f"Baris noise (ditandai processing.py) dikeluarkan: {n_noise} -> sisa {len(df):,} baris")

    # Pastikan kolom fitur lengkap
    fitur_tersedia = [c for c in FITUR_HARGA if c in df.columns]
    fitur_hilang = [c for c in FITUR_HARGA if c not in df.columns]
    if fitur_hilang:
        print(f"[PERINGATAN] Kolom fitur tidak ditemukan, dilewati: {fitur_hilang}")
    fitur_harga_aktual = fitur_tersedia
    cat_aktual = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_aktual = [c for c in NUMERICAL_COLS if c in df.columns]

    # Buat preprocessor dengan kolom yang tersedia
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_aktual),
        ('num', StandardScaler(), num_aktual),
    ])

    # Drop baris dengan fitur atau harga kosong
    df_model = df.dropna(subset=fitur_harga_aktual + ['harga']).copy()
    print(f"Baris valid untuk regresi: {len(df_model):,}")

    # ==============================================================
    # TAHAP 1: SPLIT TRAIN / TEST (SEBELUM LABEL)
    # ==============================================================
    print("\n" + "=" * 75)
    print("TAHAP 1: SPLIT TRAIN / TEST (sebelum label dibuat)")
    print("=" * 75)

    # Stratifikasi berdasarkan merek agar representasi merata
    # Gabungkan merek bersampel kecil ke kategori "Lainnya" untuk stratifikasi
    merek_counts = df_model['merek'].value_counts()
    merek_minor = merek_counts[merek_counts < 10].index
    strat_col = df_model['merek'].where(~df_model['merek'].isin(merek_minor), 'Lainnya')

    idx_train, idx_test = train_test_split(
        df_model.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=strat_col)

    df_model['split'] = 'test'
    df_model.loc[idx_train, 'split'] = 'train'

    df_train = df_model.loc[idx_train].copy()
    df_test = df_model.loc[idx_test].copy()
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    # ==============================================================
    # TAHAP 2: PERBANDINGAN MODEL REGRESI HARGA WAJAR
    # ==============================================================
    print("\n" + "=" * 75)
    print("TAHAP 2: PERBANDINGAN MODEL REGRESI HARGA WAJAR (target log-harga)")
    print("=" * 75)

    X_train_reg = df_train[fitur_harga_aktual]
    y_train_reg = df_train['harga']
    y_train_log = np.log1p(y_train_reg)

    kandidat_regresi = {
        "Linear Regression": Pipeline([('prep', prep), ('reg', LinearRegression())]),
        "Ridge (alpha=1.0)": Pipeline([('prep', prep), ('reg', Ridge(alpha=1.0))]),
        "Gradient Boosting": Pipeline([
            ('prep', prep),
            ('reg', GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)),
        ]),
    }

    kf = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    hasil_regresi = []

    print(f"\n{'Model':25s} | {'MAE':>15s} | {'RMSE':>15s} | {'MAPE%':>8s} | {'R²':>8s}")
    print("-" * 80)

    oof_preds = {}  # simpan OOF predictions per model

    for nama, pipe in kandidat_regresi.items():
        # OOF prediction pada log-harga, lalu konversi balik
        oof_log = cross_val_predict(pipe, X_train_reg, y_train_log, cv=kf, n_jobs=-1)
        oof_harga = np.expm1(oof_log)
        oof_harga = np.clip(oof_harga, 1_000_000, None)
        oof_preds[nama] = oof_harga

        mae = mean_absolute_error(y_train_reg, oof_harga)
        mse = mean_squared_error(y_train_reg, oof_harga)
        rmse_val = rmse(y_train_reg, oof_harga)
        mape = mean_absolute_percentage_error(y_train_reg, oof_harga) * 100
        r2 = r2_score(y_train_reg, oof_harga)

        hasil_regresi.append({
            "Model": nama, "MAE": mae, "MSE": mse, "RMSE": rmse_val,
            "MAPE_pct": mape, "R2": r2,
        })
        print(f"  {nama:23s} | Rp {mae:>12,.0f} | Rp {rmse_val:>12,.0f} | {mape:>7.2f} | {r2:>7.4f}")

    df_hasil_reg = pd.DataFrame(hasil_regresi)
    df_hasil_reg.to_csv(os.path.join(OUTPUT_DIR, "perbandingan_model_regresi.csv"), index=False)

    # Pilih model terbaik berdasarkan MAPE terendah
    best_reg_name = df_hasil_reg.sort_values("MAPE_pct").iloc[0]["Model"]
    best_pipe = kandidat_regresi[best_reg_name]
    best_oof = oof_preds[best_reg_name]
    print(f"\n  Model regresi terpilih: {best_reg_name} (MAPE terendah)")

    # --- Analisis Residual ---
    print("\n  --- Analisis Residual ---")
    residual = y_train_reg.values - best_oof
    print(f"  Mean residual     : Rp {residual.mean():>15,.0f}")
    print(f"  Std residual      : Rp {residual.std():>15,.0f}")
    print(f"  Median residual   : Rp {np.median(residual):>15,.0f}")
    print(f"  Residual positif  : {(residual > 0).sum():,} ({(residual > 0).mean()*100:.1f}%)")
    print(f"  Residual negatif  : {(residual < 0).sum():,} ({(residual < 0).mean()*100:.1f}%)")

    # Grafik residual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(best_oof, residual, alpha=0.3, s=10, color="#2563EB")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Harga Wajar Prediksi (Rp)")
    axes[0].set_ylabel("Residual (Rp)")
    axes[0].set_title(f"Residual Plot — {best_reg_name}")
    axes[1].hist(residual / 1e6, bins=50, color="#059669", edgecolor="white")
    axes[1].set_xlabel("Residual (Juta Rp)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].set_title("Distribusi Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_analisis_residual.png"), dpi=200)
    plt.close(fig)

    # ==============================================================
    # TAHAP 3: HARGA WAJAR (OOF train + predict test)
    # ==============================================================
    print("\n" + "=" * 75)
    print("TAHAP 3: HARGA WAJAR (OOF untuk train, model penuh untuk test)")
    print("=" * 75)

    # --- Harga wajar training (OOF) ---
    df_train['harga_wajar_model'] = np.clip(best_oof, 1_000_000, None)

    # Fallback untuk kategori bersampel sedikit (leave-one-out)
    sampel_lain = df_train['jumlah_sampel_kategori'] - 1
    hw_final_train, sumber_train = [], []
    for idx, r in df_train.iterrows():
        hw, sm = float(r['harga_wajar_model']), "model ML"
        if sampel_lain[idx] < AMBANG_KEYAKINAN_SAMPEL:
            hf, s = hitung_harga_wajar_fallback(
                df_train, r['merek'], r['model'], r['usia_mobil'], exclude_idx=idx)
            if hf is not None:
                hw, sm = hf, s
        hw_final_train.append(hw)
        sumber_train.append(sm)
    df_train['harga_wajar_final'] = hw_final_train
    df_train['sumber_harga_wajar'] = sumber_train

    # --- Latih model dari SELURUH data training ---
    model_final = TransformedTargetRegressor(
        regressor=best_pipe, func=np.log1p, inverse_func=np.expm1
    )
    model_final.fit(df_train[fitur_harga_aktual], df_train['harga'])

    # --- Harga wajar test (predict langsung, bukan OOF, langsung menghasilkan Rupiah) ---
    pred_test = model_final.predict(df_test[fitur_harga_aktual])
    df_test['harga_wajar_model'] = np.clip(pred_test, 1_000_000, None)

    # Fallback test (tanpa leave-one-out, pakai seluruh training sebagai referensi)
    hw_final_test, sumber_test = [], []
    for idx, r in df_test.iterrows():
        hw, sm = float(r['harga_wajar_model']), "model ML"
        sampel = int((df_train[(df_train['merek'] == r['merek']) &
                               (df_train['model'] == r['model']) &
                               (df_train['usia_bracket'] == r['usia_bracket'])].shape[0]))
        if sampel < AMBANG_KEYAKINAN_SAMPEL:
            hf, s = hitung_harga_wajar_fallback(
                df_train, r['merek'], r['model'], r['usia_mobil'])
            if hf is not None:
                hw, sm = hf, s
        hw_final_test.append(hw)
        sumber_test.append(sm)
    df_test['harga_wajar_final'] = hw_final_test
    df_test['sumber_harga_wajar'] = sumber_test

    # Metrik harga wajar
    for bagian, df_b in [("Train (OOF)", df_train), ("Test", df_test)]:
        mape_b = mean_absolute_percentage_error(df_b['harga'], df_b['harga_wajar_final']) * 100
        mae_b = mean_absolute_error(df_b['harga'], df_b['harga_wajar_final'])
        rmse_b = rmse(df_b['harga'], df_b['harga_wajar_final'])
        r2_b = r2_score(df_b['harga'], df_b['harga_wajar_final'])
        print(f"  {bagian:15s}: MAPE {mape_b:.2f}% | MAE Rp {mae_b:,.0f} | RMSE Rp {rmse_b:,.0f} | R² {r2_b:.4f}")

    # ==============================================================
    # TAHAP 4: LABEL PROXY (TERTILE — hanya dari data training)
    # ==============================================================
    print("\n" + "=" * 75)
    print("TAHAP 4: LABEL PROXY LIKUIDITAS (TERTILE)")
    print("=" * 75)

    # Deviasi harga
    df_train['deviasi_persen'] = klip_deviasi(
        (df_train['harga'] - df_train['harga_wajar_final']) / df_train['harga_wajar_final'] * 100)
    df_test['deviasi_persen'] = klip_deviasi(
        (df_test['harga'] - df_test['harga_wajar_final']) / df_test['harga_wajar_final'] * 100)

    # Ambang tertile HANYA dari data training (P12 slide 7 & 24)
    q33 = float(df_train['deviasi_persen'].quantile(0.33))
    q67 = float(df_train['deviasi_persen'].quantile(0.67))
    print(f"  Ambang tertile (HANYA dari train): Q33 = {q33:.2f}% | Q67 = {q67:.2f}%")

    # Terapkan label ke kedua set
    for df_bagian in [df_train, df_test]:
        df_bagian['status_likuiditas'] = np.where(
            df_bagian['deviasi_persen'] <= q33, "Cepat",
            np.where(df_bagian['deviasi_persen'] <= q67, "Sedang", "Lambat"))

        # Penyesuaian bisnis: km tinggi + tanpa garansi → turunkan dari Cepat ke Sedang
        turun = ((df_bagian['status_likuiditas'] == "Cepat") &
                 (df_bagian['km_per_tahun'] > 30000) &
                 (df_bagian['ada_garansi'] == 'Tidak'))
        df_bagian.loc[turun, 'status_likuiditas'] = "Sedang"

    print("\n  Distribusi label (Train):")
    print(df_train['status_likuiditas'].value_counts().reindex(LABEL_ORDER).to_string())
    print("\n  Distribusi label (Test):")
    print(df_test['status_likuiditas'].value_counts().reindex(LABEL_ORDER).to_string())

    print("\n  [CATATAN SIRKULARITAS] Label dibentuk DARI deviasi_persen (+ penyesuaian "
          "km/garansi). Karena deviasi_persen nanti juga dipakai sbg fitur classifier, "
          "akurasi tinggi sebagian besar wajar terjadi. Lihat uji ablation di "
          "train_model_klasifikasi.py.")

    # ==============================================================
    # SIMPAN KELUARAN
    # ==============================================================
    print("\n" + "=" * 75)
    print("SIMPAN KELUARAN")
    print("=" * 75)

    # Gabungkan kembali train + test
    df_berlabel = pd.concat([df_train, df_test], axis=0).sort_index()

    # Simpan dataset berlabel
    df_berlabel.to_csv(DATASET_BERLABEL, index=False, encoding='utf-8-sig')
    print(f"  Dataset berlabel: {DATASET_BERLABEL} ({len(df_berlabel):,} baris)")

    # Simpan model regresi
    with open(MODEL_REGRESI_FILE, "wb") as f:
        pickle.dump(model_final, f)
    print(f"  Model regresi   : {MODEL_REGRESI_FILE}")

    # Simpan metadata
    mape_train = mean_absolute_percentage_error(df_train['harga'], df_train['harga_wajar_final']) * 100
    mae_train = mean_absolute_error(df_train['harga'], df_train['harga_wajar_final'])
    mse_train = mean_squared_error(df_train['harga'], df_train['harga_wajar_final'])
    rmse_train = rmse(df_train['harga'], df_train['harga_wajar_final'])
    r2_train = r2_score(df_train['harga'], df_train['harga_wajar_final'])

    meta = {
        "q33": q33,
        "q67": q67,
        "label_order": LABEL_ORDER,
        "fitur_harga": fitur_harga_aktual,
        "categorical_cols": cat_aktual,
        "numerical_cols": num_aktual,
        "ambang_sampel": AMBANG_KEYAKINAN_SAMPEL,
        "model_regresi": best_reg_name,
        "mape_harga_wajar_pct": round(mape_train, 2),
        "mae_harga_wajar_rupiah": round(mae_train, 0),
        "mse_harga_wajar": round(mse_train, 0),
        "rmse_harga_wajar_rupiah": round(rmse_train, 0),
        "r2_harga_wajar": round(r2_train, 4),
        "train_size": len(df_train),
        "test_size": len(df_test),
        "tanggal_label": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata         : {META_FILE}")

    # --- Visualisasi ---
    # Grafik perbandingan model regresi
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_hasil_reg))
    ax.bar(x - 0.2, df_hasil_reg["MAPE_pct"], 0.4, label="MAPE (%)", color="#2563EB")
    ax.bar(x + 0.2, df_hasil_reg["R2"] * 100, 0.4, label="R² × 100", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(df_hasil_reg["Model"], rotation=15, ha="right")
    ax.set_title("Perbandingan Model Regresi Harga Wajar")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_perbandingan_regresi.png"), dpi=200)
    plt.close(fig)

    # Grafik distribusi label
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (nama, df_b) in zip(axes, [("Train", df_train), ("Test", df_test)]):
        cnt = df_b['status_likuiditas'].value_counts().reindex(LABEL_ORDER)
        ax.bar(cnt.index, cnt.values, color=["#059669", "#F59E0B", "#DC2626"])
        ax.set_title(f"Distribusi Label — {nama}")
        ax.set_ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_distribusi_label_train_test.png"), dpi=200)
    plt.close(fig)

    # Grafik deviasi per label
    fig, ax = plt.subplots(figsize=(8, 5))
    data_box = [df_train.loc[df_train['status_likuiditas'] == l, 'deviasi_persen'] for l in LABEL_ORDER]
    bp = ax.boxplot(data_box, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(LABEL_ORDER) + 1))
    ax.set_xticklabels(LABEL_ORDER)
    for patch, warna in zip(bp['boxes'], ["#059669", "#F59E0B", "#DC2626"]):
        patch.set_facecolor(warna)
    ax.axhline(q33, color="gray", linestyle="--", linewidth=1, label=f"Q33 ({q33:.1f}%)")
    ax.axhline(q67, color="gray", linestyle=":", linewidth=1, label=f"Q67 ({q67:.1f}%)")
    ax.set_ylabel("Deviasi Harga (%)")
    ax.set_title("Sebaran Deviasi Harga per Label (data training)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_deviasi_per_label.png"), dpi=200)
    plt.close(fig)

    print(f"\n  Grafik disimpan di '{OUTPUT_DIR}/'.")
    print("\n[SELESAI] Pelabelan selesai. Lanjutkan ke analisis_fitur.py.")


if __name__ == "__main__":
    main()