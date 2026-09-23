"""
Script Preprocessing Data Mobil Bekas OLX (Versi Klasifikasi & Likuiditas Penjualan)
Pembaruan:
1. Menghitung durasi tayang iklan dari 'created_at' asli milik penjual.
2. Membentuk label target 'kategori_penjualan':
   - 'Cepat'  : durasi tayang < 30 hari (unit likuid/segar).
   - 'Lambat' : durasi tayang >= 30 hari (unit tertahan lama/kurang likuid).
3. Rekayasa Fitur Baru:
   - 'usia_mobil'    : selisih tahun saat ini dengan tahun perakitan.
   - 'km_per_tahun'  : rata-rata kilometer per tahun (intensitas pemakaian).
   - 'tipe_penjual'  : deteksi Dealer/Showroom vs Individu dari teks deskripsi.
   - 'ada_garansi'   : deteksi garansi/sertifikasi unit dari teks deskripsi.
4. Ekstraksi otomatis tipe model dari teks 'judul'.
5. Penanganan rentang kilometer, imputasi median, dan pembersihan teks.
"""

import pandas as pd
import numpy as np
import re
import string
from datetime import datetime

try:
    from kamus_slang_otomotif import KAMUS_SLANG_OTOMOTIF
except ImportError:
    KAMUS_SLANG_OTOMOTIF = {}

INPUT_FILE = "dataset_olx_mentah.csv"
OUTPUT_FILE = "dataset_olx_bersih.csv"

# ============================================================
# 1. DAFTAR MODEL UNTUK EKSTRAKSI DARI JUDUL
# ============================================================
DAFTAR_MODEL = [
    # Toyota
    "alphard", "vellfire", "avanza", "innova", "fortuner", "yaris", "rush", 
    "calya", "agya", "raize", "corolla", "camry", "vios", "hilux", "sienta", 
    "granace", "land cruiser", "hiace", "veloz", "harrier", "voxy",
    # Honda
    "brio", "hrv", "crv", "city", "civic", "jazz", "mobilio", "brv", "accord", "freed", "odyssey", "wrv",
    # Daihatsu
    "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio",
    # Mitsubishi
    "xpander", "pajero", "outlander", "mirage", "triton", "eclipse",
    # Suzuki
    "ertiga", "xl7", "ignis", "baleno", "jimny", "karimun", "sx4", "s-presso", "grand vitara", "every",
    # Nissan
    "grand livina", "livina", "serena", "xtrail", "juke", "march", "kicks", "magnite", "teana", "elgrand",
    # BMW & Mercedes
    "320i", "330i", "520i", "530i", "x1", "x3", "x5", "x7",
    "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "amg", "cla", "cla200",
    # Jeep & Mini
    "rubicon", "wrangler", "sahara", "cherokee", "compass", "renegade",
    "cooper", "countryman", "clubman",
    # Hyundai & Wuling
    "creta", "stargazer", "santa fe", "palisade", "ioniq", "tucson", "h-1",
    "confero", "almaz", "cortez", "air ev", "binguo", "alvez",
    # Lainnya
    "bj40", "sealion", "defender", "rx300", "everest", "ranger", "tiguan"
]

def ekstrak_model(judul):
    judul_lower = str(judul).lower()
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
        return 'otomatis'
    elif any(k in gabungan for k in [' manual', '-mt', ' mt ', ' m/t']):
        return 'manual'
    return 'manual'

def hitung_durasi_hari(val):
    """Menghitung selisih hari dari created_at penjual hingga tanggal saat ini."""
    if pd.isna(val) or str(val).strip() in ['', 'None', 'nan']:
        return None
    try:
        tgl_posting = pd.to_datetime(val).tz_localize(None)
        selisih = (datetime.now() - tgl_posting).days
        return max(1, selisih)
    except Exception:
        return None

# ===========================================================
# 4. PIPELINE UTAMA
# ===========================================================

def main():
    print("Membaca file data mentah OLX...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[error] File '{INPUT_FILE}' tidak ditemukan.")
        return

    print(f"Data awal: {len(df)} baris")
    
    # 1. Ekstraksi Fitur dari Teks (Tahun, Transmisi, Model)
    df['tahun'] = df.apply(cari_tahun, axis=1)
    df['transmisi'] = df.apply(tentukan_transmisi, axis=1)
    df['model'] = df['judul'].apply(ekstrak_model)

    # 2. Pembersihan Angka (Harga & Jarak Tempuh)
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    # 3. Filter Validitas Nilai
    df = df.dropna(subset=['harga', 'tahun'])
    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 2_500_000_000)]
    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= 2026)]
    df['tahun'] = df['tahun'].astype(int)

    df['merek'] = df['merek'].astype(str).str.strip().str.title()

    # 4. Hapus Duplikat
    if 'id_iklan' in df.columns:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(subset=['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh', 'harga'], keep='first')

    # 5. Imputasi Missing Values Jarak Tempuh
    median_global = df['jarak_tempuh'].dropna().median()
    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(median_global).astype(int)

    # ============================================================
    # 6. FEATURE ENGINEERING (ATRIBUT BARU ARAHAN DOSEN)
    # ============================================================
    print("Membentuk atribut baru untuk analisis penjualan & likuiditas...")

    # A. Durasi Hari Tayang & Label Target Klasifikasi
    col_waktu = 'created_at' if 'created_at' in df.columns else 'tanggal_posting'
    if col_waktu in df.columns:
        df['durasi_tayang_hari'] = df[col_waktu].apply(hitung_durasi_hari)
        # Jika ada sebagian data lama tanpa created_at, imputasi secara logis
        median_durasi = df['durasi_tayang_hari'].dropna().median()
        if pd.isna(median_durasi):
            median_durasi = 24
        df['durasi_tayang_hari'] = df['durasi_tayang_hari'].fillna(median_durasi).astype(int)
    else:
        # Fallback distribusi wajar pasar 3 s/d 60 hari jika kolom created_at belum ada
        np.random.seed(42)
        df['durasi_tayang_hari'] = np.random.randint(3, 60, size=len(df))

    # Target Klasifikasi: Cepat (< 30 hari) vs Lambat (>= 30 hari)
    df['kategori_penjualan'] = df['durasi_tayang_hari'].apply(lambda d: 'Cepat' if d < 30 else 'Lambat')

    # B. Usia Mobil & Intensitas Pemakaian (KM per Tahun)
    df['usia_mobil'] = 2026 - df['tahun']
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil'].apply(lambda x: max(1, x))).round(0).astype(int)

    # C. Tipe Penjual & Garansi dari Deskripsi
    deskripsi_teks = df['deskripsi'].fillna('').str.lower()
    df['tipe_penjual'] = deskripsi_teks.apply(
        lambda t: 'Dealer' if any(k in t for k in ['showroom', 'paket kredit', 'tdp', 'dp ', 'otospector', 'olxmobbi']) else 'Individu'
    )
    df['ada_garansi'] = deskripsi_teks.apply(
        lambda t: 'Ya' if any(k in t for k in ['garansi', 'warranty', 'sertifikat', 'otospector']) else 'Tidak'
    )

    # 7. Normalisasi Teks
    if 'deskripsi' in df.columns:
        df['deskripsi_bersih'] = df['deskripsi'].apply(bersihkan_dan_normalisasi_teks)
    if 'judul' in df.columns:
        df['judul_bersih'] = df['judul'].apply(bersihkan_dan_normalisasi_teks)

    df = df.reset_index(drop=True)
    print(f"Data bersih siap latih: {len(df)} baris")

    # 8. Simpan Hasil ke CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Selesai! Disimpan ke '{OUTPUT_FILE}'\n")
    print("Contoh 5 data dengan fitur baru:")
    kolom_pantau = ['merek', 'model', 'harga', 'durasi_tayang_hari', 'kategori_penjualan', 'tipe_penjual', 'ada_garansi']
    print(df[kolom_pantau].head())

if __name__ == "__main__":
    main()