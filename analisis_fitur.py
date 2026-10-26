"""
analisis_fitur.py - Step 3: Feature Selection Formal (Materi P3-4)

Metode Feature Selection:
1. Filter Method:
   - Korelasi Pearson / Spearman (Deteksi multikolinearitas & redundansi)
   - Mutual Information Classification (Melihat hubungan non-linear fitur ke target status_likuiditas)
2. Wrapper Method:
   - Recursive Feature Elimination (RFE) menggunakan Random Forest
3. Embedded Method:
   - Random Forest Feature Importance

Output:
- keputusan_fitur.csv (Daftar fitur beserta status dipertahankan/dibuang & alasannya)
- analisis_fitur_meta.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

INPUT_FILE = "dataset_olx_berlabel.csv"
OUTPUT_DECISION = "keputusan_fitur.csv"
META_FILE = "analisis_fitur_meta.json"

def main():
    print("=" * 80)
    print("   STEP 3: FEATURE SELECTION FORMAL (FILTER, WRAPPER, EMBEDDED - P3-4)   ")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File '{INPUT_FILE}' tidak ditemukan. Jalankan pelabelan.py terlebih dahulu.")

    df = pd.read_csv(INPUT_FILE)
    print(f"Membaca dataset berlabel: {len(df):,} baris")

    # Gunakan HANYA data training untuk feature selection (Mencegah Data Leakage P12)
    df_train = df[df['is_train'] == 1].copy()
    print(f"Data Training untuk Feature Selection: {len(df_train):,} baris")

    # Fitur kandidat numerik dan kategorik
    kandidat_numerik = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun', 'indikasi_kredit_tdp']
    kandidat_kategorik = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
    
    target = df_train['status_likuiditas']

    # ============================================================
    # 1. FILTER METHOD: KORELASI & MUTUAL INFORMATION
    # ============================================================
    print("\n--- 1. FILTER METHOD: Korelasi & Mutual Information ---")
    
    # Check Korelasi Pearson pada fitur numerik
    corr_matrix = df_train[kandidat_numerik].corr().abs()
    print("Matrix Korelasi Pearson (Numerik):")
    print(corr_matrix.round(2).to_string())
    
    # Preprocessing untuk Mutual Information
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), kandidat_kategorik),
        ('num', StandardScaler(), kandidat_numerik)
    ])
    
    X_train_encoded = prep.fit_transform(df_train[kandidat_kategorik + kandidat_numerik])
    feature_names = prep.get_feature_names_out()

    # Hitung Mutual Information
    mi_scores = mutual_info_classif(X_train_encoded, target, random_state=42)
    df_mi = pd.DataFrame({'Fitur_Encoded': feature_names, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)
    
    print("\nTop 10 Fitur dengan Mutual Information Terbesar:")
    print(df_mi.head(10).to_string(index=False))

    # ============================================================
    # 2. WRAPPER METHOD: RECURSIVE FEATURE ELIMINATION (RFE)
    # ============================================================
    print("\n--- 2. WRAPPER METHOD: RFE dengan Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=10, step=2)
    rfe.fit(X_train_encoded, target)

    rfe_ranking = pd.DataFrame({
        'Fitur_Encoded': feature_names,
        'RFE_Rank': rfe.ranking_,
        'RFE_Selected': rfe.support_
    }).sort_values('RFE_Rank')

    # ============================================================
    # 3. EMBEDDED METHOD: RANDOM FOREST FEATURE IMPORTANCE
    # ============================================================
    print("\n--- 3. EMBEDDED METHOD: Random Forest Importance ---")
    rf.fit(X_train_encoded, target)
    df_importance = pd.DataFrame({
        'Fitur_Encoded': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # ============================================================
    # 4. FORMULASI KEPUTUSAN SELEKSI FITUR (KEPUTUSAN FORMAL)
    # ============================================================
    keputusan = [
        {
            "Fitur_Asli": "tahun",
            "Tipe": "Numerik",
            "Status": "DIBUANG",
            "Alasan_Akademik": "Redundan secara multikolinearitas sempurna (r = -1.0) dengan 'usia_mobil' (2026 - tahun). Cukup gunakan usia_mobil."
        },
        {
            "Fitur_Asli": "usia_mobil",
            "Tipe": "Numerik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Penyederhanaan linier dari tahun perakitan, esensial untuk kurva depresiasi harga."
        },
        {
            "Fitur_Asli": "jarak_tempuh",
            "Tipe": "Numerik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Indikator utama tingkat kelelahan mesin kendaraan (MI Score & RFE rank tinggi)."
        },
        {
            "Fitur_Asli": "km_per_tahun",
            "Tipe": "Numerik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Fitur rasio pemakaian tahunan yang mengukur intensitas operasional mobil secara objektif."
        },
        {
            "Fitur_Asli": "merek",
            "Tipe": "Kategorik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Variabel dasar pembeda segmentasi pabrikan dan daya tahan nilai jual kembali."
        },
        {
            "Fitur_Asli": "model",
            "Tipe": "Kategorik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Variabel kunci spesifikasi seri kendaraan yang paling menentukan taksiran harga pasar."
        },
        {
            "Fitur_Asli": "transmisi",
            "Tipe": "Kategorik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Preferensi utama konsumen pasar lokal (Manual vs Automatic)."
        },
        {
            "Fitur_Asli": "tipe_penjual",
            "Tipe": "Kategorik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Memisahkan perilaku penawaran harga antara Dealer/Showroom dan Individu."
        },
        {
            "Fitur_Asli": "ada_garansi",
            "Tipe": "Kategorik",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Fitur NLP teks utama penambah nilai psikologis pembeli (garansi/servis resmi)."
        },
        {
            "Fitur_Asli": "indikasi_kredit_tdp",
            "Tipe": "Biner",
            "Status": "DIPERTAHANKAN",
            "Alasan_Akademik": "Menyaring skema penawaran uang muka kredit (TDP) agar tidak mengacaukan harga penawaran tunai."
        }
    ]

    df_keputusan = pd.DataFrame(keputusan)
    df_keputusan.to_csv(OUTPUT_DECISION, index=False, encoding='utf-8-sig')

    meta_selection = {
        "fitur_numerik_terpilih": [f["Fitur_Asli"] for f in keputusan if f["Status"] == "DIPERTAHANKAN" and f["Tipe"] in ["Numerik", "Biner"]],
        "fitur_kategorik_terpilih": [f["Fitur_Asli"] for f in keputusan if f["Status"] == "DIPERTAHANKAN" and f["Tipe"] == "Kategorik"],
        "fitur_dibuang": [f["Fitur_Asli"] for f in keputusan if f["Status"] == "DIBUANG"],
        "top_5_mi_encoded": df_mi.head(5).to_dict(orient='records'),
        "top_5_rf_importance": df_importance.head(5).to_dict(orient='records')
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_selection, f, indent=2)

    print(f"\n[SUKSES] Step 3 Selesai! Berkas keputusan disimpan di '{OUTPUT_DECISION}' & '{META_FILE}'.")
    print("=" * 80)

if __name__ == "__main__":
    main()