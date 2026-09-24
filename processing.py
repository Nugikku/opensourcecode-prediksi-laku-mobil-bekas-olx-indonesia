"""
Script Preprocessing Data Mobil Bekas OLX (AutoLiquid DSS Hybrid)
Menggabungkan keandalan ekstraksi script lama dengan fitur analitis DSS.
"""

import os
import re
import string
import pandas as pd
import numpy as np

try:
    from kamus_slang_otomotif import KAMUS_SLANG_OTOMOTIF
except ImportError:
    KAMUS_SLANG_OTOMOTIF = {}

INPUT_FILE = "dataset_olx_mentah.csv"
OUTPUT_FILE = "dataset_olx_bersih.csv"
TAHUN_SEKARANG = 2026

# ============================================================
# 1. DAFTAR MODEL UNTUK EKSTRAKSI DARI JUDUL
# ============================================================
DAFTAR_MODEL = [
    # Toyota
    "alphard", "vellfire", "avanza", "innova", "fortuner", "yaris", "rush", 
    "calya", "agya", "raize", "corolla", "camry", "vios", "hilux", "sienta", 
    "granace", "land cruiser", "hiace", "veloz", "harrier", "voxy", "kijang",
    # Honda
    "brio", "hrv", "crv", "city", "civic", "jazz", "mobilio", "brv", "accord", "freed", "odyssey", "wrv",
    # Daihatsu
    "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio", "taruna",
    # Mitsubishi
    "xpander", "pajero", "outlander", "mirage", "triton", "eclipse", "l300",
    # Suzuki
    "ertiga", "xl7", "ignis", "baleno", "jimny", "karimun", "sx4", "s-presso", "grand vitara", "every", "apv",
    # Nissan
    "grand livina", "livina", "serena", "xtrail", "juke", "march", "kicks", "magnite", "teana", "elgrand", "evalia",
    # BMW & Mercedes
    "320i", "330i", "520i", "530i", "x1", "x3", "x5", "x7",
    "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "amg", "cla", "cla200",
    # Jeep & Mini
    "rubicon", "wrangler", "sahara", "cherokee", "compass", "renegade",
    "cooper", "countryman", "clubman",
    # Hyundai & Wuling
    "creta", "stargazer", "santa fe", "palisade", "ioniq", "tucson", "h-1",
    "confero", "almaz", "cortez", "air ev", "binguo", "alvez", "formo",
    # Lainnya
    "bj40", "sealion", "defender", "rx300", "everest", "ranger", "tiguan"
]

def ekstrak_model(row):
    model_asli = str(row.get('model', '')).strip()
    if model_asli and model_asli.lower() not in ['lainnya', 'others', 'nan', '']:
        return model_asli.title()
    
    judul_lower = str(row.get('judul', '')).lower()
    for m in DAFTAR_MODEL:
        if re.search(r'\b' + re.escape(m) + r'\b', judul_lower):
            return m.title()
    return "Lainnya"

# ============================================================
# 2. TEXT CLEANING & NORMALISASI
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

# ============================================================
# 3. NUMERICAL & FEATURE EXTRACTION
# ============================================================
def bersihkan_angka_harga(nilai):
    if pd.isna(nilai):
        return None
    try:
        val_float = float(nilai)
        return int(val_float)
    except (ValueError, TypeError):
        pass
    angka_saja = re.sub(r'\D', '', str(nilai))
    return int(angka_saja) if angka_saja else None

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
    if any(k in gabungan for k in [' matic', '-at', ' at ', ' a/t', 'otomatis', ' tiptronic', ' triptonic', ' cvt ']):
        return 'automatic'
    elif any(k in gabungan for k in [' manual', '-mt', ' mt ', ' m/t']):
        return 'manual'
    return 'automatic'

# ============================================================
# 4. PIPELINE UTAMA
# ============================================================
def main():
    print(f"Membaca {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"[error] File '{INPUT_FILE}' tidak ditemukan.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Data awal: {len(df):,} baris")
    
    # 1. Ekstraksi Fitur Dasar
    df['tahun'] = df.apply(cari_tahun, axis=1)
    df['transmisi'] = df.apply(tentukan_transmisi, axis=1)
    df['model'] = df.apply(ekstrak_model, axis=1)

    # 2. Pembersihan Angka
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    # 3. Filter Validitas Harga & Tahun
    df = df.dropna(subset=['harga', 'tahun'])
    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 3_000_000_000)]
    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= TAHUN_SEKARANG)]
    df['tahun'] = df['tahun'].astype(int)

    # 4. Standarisasi Format Merek
    df['merek'] = df['merek'].astype(str).str.strip().str.title()
    df = df[~df['merek'].str.lower().isin(['lainnya', 'others', 'nan', ''])]

    # 5. Imputasi Nilai Jarak Tempuh
    median_global = df['jarak_tempuh'].dropna().median()
    if pd.isna(median_global):
        median_global = 50000
    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(median_global).astype(int)

    # 6. Fitur Pendukung AutoLiquid DSS
    df['usia_mobil'] = df['tahun'].apply(lambda x: max(1, int(TAHUN_SEKARANG - x)))
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil']).round().astype(int)

    # 7. Ekstraksi Fitur Teks (Garansi & Indikasi TDP Kredit)
    if 'deskripsi' in df.columns:
        df['deskripsi_bersih'] = df['deskripsi'].apply(bersihkan_dan_normalisasi_teks)
    else:
        df['deskripsi_bersih'] = ""

    if 'judul' in df.columns:
        df['judul_bersih'] = df['judul'].apply(bersihkan_dan_normalisasi_teks)
    else:
        df['judul_bersih'] = ""

    teks_gabung = df['judul_bersih'] + " " + df['deskripsi_bersih']
    pola_garansi = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|service record|buku servis)\b'
    df['ada_garansi'] = teks_gabung.apply(lambda x: 'Ya' if re.search(pola_garansi, str(x)) else 'Tidak')

    pola_kredit = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit|kredit syariah)\b'
    df['indikasi_kredit_tdp'] = teks_gabung.apply(lambda x: 1 if re.search(pola_kredit, str(x)) else 0)

    # 8. Normalisasi Tipe Penjual
    if 'tipe_penjual_badge' in df.columns:
        df['tipe_penjual'] = df['tipe_penjual_badge'].apply(
            lambda x: 'Dealer' if pd.notna(x) and str(x).strip().lower() not in ['individu', 'none', '', 'nan'] else 'Individu'
        )
    else:
        df['tipe_penjual'] = 'Individu'

    # 9. Hapus Duplikat
    if 'id_iklan' in df.columns and df['id_iklan'].notna().sum() > 0:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(subset=['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh', 'harga'], keep='first')

    df = df.reset_index(drop=True)
    print(f"Data bersih siap training: {len(df):,} baris")
    print(f"Rentang Harga : Rp {int(df['harga'].min()):,} s/d Rp {int(df['harga'].max()):,}")
    print(f"Merek Teratas :\n{df['merek'].value_counts().head(5)}")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[SUKSES] Dataset berhasil disimpan ke '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()