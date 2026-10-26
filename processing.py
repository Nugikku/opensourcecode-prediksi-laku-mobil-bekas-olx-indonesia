"""
AutoLaku - Script Preprocessing Data Mobil Bekas OLX (Langkah 1)
Sesuai standar materi P3-4: Data Preprocessing, Cleaning, Outlier Detection, & Feature Engineering.

Output:
- dataset_olx_bersih.csv (Dataset hasil pembersihan)
- audit_preprocessing.json (Pencatatan jejak data & missing value audit)
"""

import os
import re
import json
import string
import pandas as pd
import numpy as np

try:
    from kamus_slang_otomotif import KAMUS_SLANG_OTOMOTIF
except ImportError:
    KAMUS_SLANG_OTOMOTIF = {}

INPUT_FILE = "dataset_olx_mentah.csv"
OUTPUT_FILE = "dataset_olx_bersih.csv"
AUDIT_FILE = "audit_preprocessing.json"
TAHUN_SEKARANG = 2026

BRACKET_USIA_TAHUN = 5
AMBANG_KEYAKINAN_SAMPEL = 15

# ============================================================
# 1. KAMUS & REGEX MODEL PER MEREK
# ============================================================
MODEL_PER_MEREK = {
    "toyota": [
        "alphard", "vellfire", "avanza", "innova", "fortuner", "yaris", "rush",
        "calya", "agya", "raize", "corolla", "camry", "vios", "hilux", "sienta",
        "granace", "land cruiser", "hiace", "veloz", "harrier", "voxy", "kijang",
    ],
    "honda": [
        "brio", "hrv", "hr-v", "crv", "cr-v", "city", "civic", "jazz", "mobilio",
        "brv", "br-v", "accord", "freed", "odyssey", "wrv", "wr-v",
    ],
    "daihatsu": [
        "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio", "taruna",
    ],
    "mitsubishi": [
        "xpander", "pajero", "outlander", "mirage", "triton", "eclipse", "l300",
    ],
    "suzuki": [
        "ertiga", "xl7", "ignis", "baleno", "jimny", "karimun", "sx4",
        "s-presso", "grand vitara", "every", "apv",
    ],
    "nissan": [
        "grand livina", "livina", "serena", "xtrail", "x-trail", "juke", "march",
        "kicks", "magnite", "teana", "elgrand", "evalia",
    ],
    "bmw": ["320i", "330i", "520i", "530i", "x1", "x3", "x5", "x7"],
    "mercedes-benz": [
        "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "amg", "cla", "cla200",
    ],
    "hyundai": ["creta", "stargazer", "santa fe", "palisade", "ioniq", "tucson", "h-1"],
    "wuling": ["confero", "almaz", "cortez", "air ev", "binguo", "alvez", "formo"],
    "mazda": ["mazda2", "mazda3", "cx3", "cx5", "cx8", "cx9", "biante"],
}

KAMUS_ALIAS_MODEL = {
    "inova": "innova", "innofa": "innova", "avanzaa": "avanza",
    "mazda 2": "mazda2", "mazda 3": "mazda3",
    "cx-3": "cx3", "cx-5": "cx5", "cx-8": "cx8", "cx-9": "cx9",
}

MODEL_PREFIX_AMBIGU = {
    "kijang": ["innova", "inova"],
}

def _buat_pola_regex(kata):
    dasar = r'\b' + re.escape(kata) + r'\b'
    kata_spesifik = MODEL_PREFIX_AMBIGU.get(kata)
    if kata_spesifik:
        pola_negatif = '|'.join(re.escape(k) for k in kata_spesifik)
        return dasar + rf'(?!\s*(?:{pola_negatif}))'
    return dasar

def ekstrak_model(row):
    model_asli = str(row.get('model', '')).strip()
    if model_asli and model_asli.lower() not in ['lainnya', 'others', 'nan', '']:
        return model_asli.title()

    merek_lower = str(row.get('merek', '')).strip().lower()
    judul_lower = str(row.get('judul', '')).lower()
    daftar_model = MODEL_PER_MEREK.get(merek_lower, [])

    kandidat_kata = list(daftar_model) + [
        alias for alias, baku in KAMUS_ALIAS_MODEL.items() if baku in daftar_model
    ]

    kandidat = []
    for m in kandidat_kata:
        pola = _buat_pola_regex(m)
        match = re.search(pola, judul_lower)
        if match:
            nama_baku = KAMUS_ALIAS_MODEL.get(m, m)
            kandidat.append((match.start(), nama_baku))

    if kandidat:
        kandidat.sort(key=lambda x: x[0])
        return kandidat[0][1].title()
    return "Lainnya"

# ============================================================
# 2. TEKS CLEANING & EKSTRAKSI FITUR TEKS NLP
# ============================================================
def bersihkan_dan_normalisasi_teks(teks):
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'wa\s*\d+|08\d+', 'kontak', teks)
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = re.sub(r'\s+', ' ', teks).strip()

    kata_kata = teks.split()
    kata_terfilter = [KAMUS_SLANG_OTOMOTIF.get(k, k) for k in kata_kata]
    return ' '.join(kata_terfilter)

NEGASI = {"tidak", "bukan", "tanpa", "belum", "no", "nggak", "gk", "ga", "tidakada"}
POLA_GARANSI = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|service record|buku servis|catatan servis)\b'
POLA_KREDIT = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit|kredit syariah)\b'
POLA_BANJIR_TABRAK_POSITIF = r'\b(bebas banjir|bebas tabrak|bebas kecelakaan|bebas banjir bebas tabrak|no banjir|no tabrak)\b'
POLA_BANJIR_TABRAK_NEGATIF = r'\b(bekas banjir|pernah banjir|bekas tabrak|pernah tabrak|terdampak banjir)\b'
POLA_PAJAK_HIDUP = r'\b(pajak hidup|pajak panjang|pajak on|taat pajak)\b'

def _ada_pola_tanpa_negasi(teks, pola):
    for m in re.finditer(pola, teks):
        sebelum = teks[:m.start()].split()[-2:]
        if any(w in NEGASI for w in sebelum):
            continue
        return True
    return False

def ekstrak_fitur_teks_luas(judul, deskripsi):
    judul_b = bersihkan_dan_normalisasi_teks(judul)
    desk_b = bersihkan_dan_normalisasi_teks(deskripsi)
    gabung = f"{judul_b} {desk_b}"
    
    # Deteksi fitur riwayat banjir/tabrak
    riwayat_buruk = 1 if re.search(POLA_BANJIR_TABRAK_NEGATIF, gabung) else 0
    riwayat_baik = 1 if _ada_pola_tanpa_negasi(gabung, POLA_BANJIR_TABRAK_POSITIF) else 0

    return {
        'judul_bersih': judul_b,
        'deskripsi_bersih': desk_b,
        'ada_garansi': 'Ya' if _ada_pola_tanpa_negasi(gabung, POLA_GARANSI) else 'Tidak',
        'indikasi_kredit_tdp': 1 if _ada_pola_tanpa_negasi(gabung, POLA_KREDIT) else 0,
        'pajak_hidup': 1 if _ada_pola_tanpa_negasi(gabung, POLA_PAJAK_HIDUP) else 0,
        'indikasi_bekas_banjir_tabrak': riwayat_buruk,
        'bebas_banjir_tabrak_claim': riwayat_baik
    }

# ============================================================
# 3. PEMBERSIHAN ANGKA, OUTLIER, DAN NOISE
# ============================================================
def is_harga_boneka(angka_str):
    if not angka_str or len(angka_str) < 6:
        return False
    if len(set(angka_str)) == 1:
        return True
    if len(angka_str) % 2 == 0 and angka_str == (angka_str[:2] * (len(angka_str) // 2)):
        return True
    return False

def bersihkan_angka_harga(nilai):
    if pd.isna(nilai):
        return None
    try:
        val_float = float(nilai)
        angka_saja = str(int(val_float))
    except (ValueError, TypeError):
        angka_saja = re.sub(r'\D', '', str(nilai))

    if not angka_saja:
        return None
    if is_harga_boneka(angka_saja):
        return None
    return int(angka_saja)

def perbaiki_jarak_tempuh(val):
    if pd.isna(val):
        return None
    val_str = str(val).replace('.', '').strip()
    if '-' in val_str:
        bagian = val_str.split('-')
        angka = [int(re.sub(r'\D', '', b)) for b in bagian if re.sub(r'\D', '', b)]
        return int(sum(angka) / len(angka)) if angka else None
    angka_saja = re.sub(r'\D', '', val_str)
    return int(angka_saja) if angka_saja else None

def cari_tahun(row):
    val = row.get('tahun')
    if pd.notna(val) and str(val).strip() not in ['', 'None', 'nan']:
        try:
            return int(float(val))
        except ValueError:
            pass

    judul = str(row.get('judul', ''))
    cocok = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', judul)
    if cocok:
        return int(cocok[-1])
    return None

def tentukan_transmisi(row):
    gabungan = f"{row.get('transmisi', '')} {row.get('judul', '')} {str(row.get('deskripsi', ''))[:300]}".lower()
    if any(k in gabungan for k in [' matic', '-at', ' at ', ' a/t', 'otomatis', ' tiptronic', ' cvt ']):
        return 'automatic'
    elif any(k in gabungan for k in [' manual', '-mt', ' mt ', ' m/t']):
        return 'manual'
    return 'automatic'

def hitung_usia_bracket(usia_mobil, ukuran_bracket=BRACKET_USIA_TAHUN):
    return (int(usia_mobil) // ukuran_bracket) * ukuran_bracket

def tandai_outlier_iqr_log_harga(df):
    """Materi P3-4: Deteksi Outlier menggunakan IQR pada log(harga) per merek.
    Outlier ditandai (is_outlier_harga) bukan dibuang, untuk membedakan mobil mewah valid."""
    df['is_outlier_harga'] = 0
    df['log_harga'] = np.log(df['harga'])
    
    for merek, g in df.groupby('merek'):
        if len(g) >= 10:
            q1 = g['log_harga'].quantile(0.25)
            q3 = g['log_harga'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_idx = g[(g['log_harga'] < lower_bound) | (g['log_harga'] > upper_bound)].index
            df.loc[outlier_idx, 'is_outlier_harga'] = 1
            
    df.drop(columns=['log_harga'], inplace=True)
    return df

# ============================================================
# 4. PIPELINE UTAMA PREPROCESSING
# ============================================================
def main():
    print("=" * 80)
    print("      STEP 1: PREPROCESSING DATASET OLX (STANDAR MATERI P3-4)     ")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File '{INPUT_FILE}' tidak ditemukan.")
        return

    df_raw = pd.read_csv(INPUT_FILE)
    n_awal = len(df_raw)
    print(f"Data Mentah Awal: {n_awal:,} baris")

    # Audit Missing Values Awal
    missing_awal = df_raw.isnull().sum().to_dict()

    # 1. Ekstraksi Dasar
    df = df_raw.copy()
    df['tahun'] = df.apply(cari_tahun, axis=1)
    df['transmisi'] = df.apply(tentukan_transmisi, axis=1)
    df['model'] = df.apply(ekstrak_model, axis=1)

    # 2. Pembersihan Angka & Filtering Domain
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    df = df.dropna(subset=['harga', 'tahun'])
    n_after_null_filter = len(df)

    # Filter Domain Harga & Tahun
    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 3_000_000_000)]
    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= TAHUN_SEKARANG)]
    df['tahun'] = df['tahun'].astype(int)
    n_after_domain_filter = len(df)

    # 3. Standardisasi Merek
    df['merek'] = df['merek'].astype(str).str.strip().str.title()
    df = df[~df['merek'].str.lower().isin(['lainnya', 'others', 'nan', ''])]

    # 4. Imputasi Jarak Tempuh & Penandaan (P3-4 Imputation Marking)
    df['is_km_imputed'] = df['jarak_tempuh'].isna().astype(int)
    median_global = df['jarak_tempuh'].dropna().median()
    if pd.isna(median_global):
        median_global = 50000
    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(median_global).astype(int)

    # 5. Usia & Fitur Operasional
    df['usia_mobil'] = df['tahun'].apply(lambda x: max(1, int(TAHUN_SEKARANG - x)))
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil']).round().astype(int)

    # 6. Deteksi Noise Domain (e.g., KM per tahun sangat ekstrem)
    df['is_noise_km_ekstrem'] = ((df['km_per_tahun'] > 100_000) | ((df['km_per_tahun'] < 200) & (df['usia_mobil'] > 3))).astype(int)

    # 7. Ekstraksi Fitur Teks NLP Terpadu
    fitur_teks = df.apply(lambda r: ekstrak_fitur_teks_luas(r.get('judul'), r.get('deskripsi')), axis=1, result_type='expand')
    for kolom in fitur_teks.columns:
        df[kolom] = fitur_teks[kolom]

    # 8. Normalisasi Tipe Penjual
    if 'tipe_penjual_badge' in df.columns:
        df['tipe_penjual'] = df['tipe_penjual_badge'].apply(
            lambda x: 'Dealer' if pd.notna(x) and str(x).strip().lower() not in ['individu', 'none', '', 'nan'] else 'Individu'
        )
    else:
        df['tipe_penjual'] = 'Individu'

    # 9. Deduplikasi
    if 'id_iklan' in df.columns and df['id_iklan'].notna().sum() > 0:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(subset=['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh', 'harga'], keep='first')
    df = df.reset_index(drop=True)
    n_akhir = len(df)

    # 10. Indikator Kelangkaan Sampel & Outlier IQR
    df['usia_bracket'] = df['usia_mobil'].apply(hitung_usia_bracket)
    df['jumlah_sampel_kategori'] = df.groupby(['merek', 'model', 'usia_bracket'])['model'].transform('count')
    df = tandai_outlier_iqr_log_harga(df)

    # Save Output Dataset
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # Audit Metadata JSON (Untuk Laporan P3-4 / Jawaban UTS)
    audit_data = {
        "jejak_data_baris": {
            "mentah_awal": n_awal,
            "setelah_filter_null": n_after_null_filter,
            "setelah_filter_domain_harga_tahun": n_after_domain_filter,
            "bersih_akhir": n_akhir
        },
        "missing_value_audit_awal": missing_awal,
        "imputasi_summary": {
            "jarak_tempuh_diimputasi": int(df['is_km_imputed'].sum()),
            "persentase_imputasi_km": round(float(df['is_km_imputed'].mean() * 100), 2)
        },
        "outlier_dan_noise_summary": {
            "jumlah_outlier_harga_iqr": int(df['is_outlier_harga'].sum()),
            "jumlah_noise_km_ekstrem": int(df['is_noise_km_ekstrem'].sum())
        },
        "catatan_metodologi_p3_4": {
            "smoothing": "Tidak diterapkan karena data bersifat tabular cross-sectional, bukan time-series bertautan.",
            "data_integration": "Tidak dilakukan karena data bersumber tunggal dari OLX Indonesia.",
            "outlier_handling": "Outlier ditandai (is_outlier_harga) dengan IQR pada log(harga) per merek, bukan dihapus, untuk mempertahankan kendaraan mewah/unik yang valid."
        }
    }

    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print("\n[SUKSES] Step 1 Selesai!")
    print(f"Data Akhir     : {n_akhir:,} baris dari {n_awal:,} data mentah awal")
    print(f"Output File    : '{OUTPUT_FILE}' & '{AUDIT_FILE}'")
    print("=" * 80)

if __name__ == "__main__":
    main()