"""
Script Preprocessing Data Mobil Bekas OLX (Versi DSS Likuiditas & Ground Truth Penjualan)
Pembaruan:
1. Perluasan kamus model kendaraan untuk meminimalkan label 'Lainnya'.
2. Pembentukan label target 'kategori_penjualan' realistis (Cepat, Sedang, Lambat)
   berdasarkan sinyal terjual, durasi mengambang di pasar, dan daya saing harga pasar.
3. Feature Engineering: usia kendaraan, intensitas km/tahun, deviasi harga pasar, 
   profil penjual, dan status garansi dari NLP teks deskripsi.
4. Pembersihan nilai kosong dan penanganan inkonsistensi teks.
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
# 1. DAFTAR MODEL KENDARAAN LENGKAP
# ============================================================
DAFTAR_MODEL = [
    # Toyota
    "zenix", "innova zenix", "innova", "avanza", "veloz", "fortuner", "alphard", "vellfire", 
    "yaris cross", "yaris", "rush", "calya", "agya", "raize", "corolla cross", "corolla altis", 
    "corolla", "camry", "vios", "hilux", "sienta", "hiace", "land cruiser", "voxy", "harrier", 
    "kijang", "crown", "granace",
    # Honda
    "brio", "hrv", "hr-v", "crv", "cr-v", "city hatchback", "city", "civic", "jazz", "mobilio", 
    "brv", "br-v", "wrv", "wr-v", "accord", "freed", "odyssey",
    # Mazda
    "cx-5", "cx5", "cx-3", "cx3", "cx-8", "cx8", "cx-9", "cx9", "cx-30", "cx30", "cx-60", "cx60",
    "mazda 2", "mazda2", "mazda 3", "mazda3", "mazda 6", "mazda6", "biante",
    # Daihatsu
    "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio", "taruna",
    # Mitsubishi
    "xpander cross", "xpander", "pajero sport", "pajero", "xforce", "outlander", "mirage", "triton", "eclipse cross", "eclipse",
    # Suzuki
    "ertiga", "xl7", "xl-7", "ignis", "baleno", "jimny", "karimun wagon", "karimun", "sx4", "s-cross", 
    "s-presso", "spresso", "grand vitara", "vitara", "apv", "carry",
    # Hyundai & Wuling
    "creta", "stargazer", "santa fe", "palisade", "ioniq 5", "ioniq 6", "ioniq", "tucson", "staria", "h-1",
    "air ev", "binguo ev", "binguo", "cloud ev", "almaz", "cortez", "confero", "alvez", "formo",
    # Nissan
    "magnite", "kicks", "livina", "grand livina", "serena", "xtrail", "x-trail", "juke", "march", "teana", "elgrand",
    # Merek Listrik & Baru (BYD, Chery, GWM, DFSK)
    "seal", "atto 3", "atto", "dolphin", "m6", "omoda 5", "omoda", "tiggo 5x", "tiggo 7", "tiggo 8", "tiggo", 
    "tank 500", "haval", "glory 580", "glory",
    # Isuzu, Ford, Chevrolet
    "panther", "mu-x", "d-max", "everest", "ranger", "ecosport", "trax", "trailblazer", "captiva", "spin",
    # Merek Eropa & Mewah
    "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "cla", "amg",
    "320i", "330i", "520i", "530i", "116i", "x1", "x3", "x5", "x7",
    "cooper", "countryman", "clubman", "rx270", "rx300", "rx350", "lm350", "brz", "wrx", "forester",
    "q5", "q7", "macan", "cayenne", "boxster", "718", "d9", "bj40", "dashing"
]

def ekstrak_model(judul):
    judul_lower = str(judul).lower()
    for m in DAFTAR_MODEL:
        pola = r'\b' + re.escape(m) + r'\b'
        if re.search(pola, judul_lower):
            return m.title()
    return "Lainnya"

# ============================================================
# 2. PEMBERSIHAN TEKS & KAMUS SLANG
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
# 3. PEMBERSIHAN FITUR NUMERIK
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
    if pd.isna(val) or str(val).strip() in ['', 'None', 'nan']:
        return None
    try:
        tgl_posting = pd.to_datetime(val).tz_localize(None)
        selisih = (datetime.now() - tgl_posting).days
        return max(1, selisih)
    except Exception:
        return None

# ============================================================
# 4. PIPELINE PREPROCESSING UTAMA
# ============================================================
def main():
    print("Membaca file data mentah OLX...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File '{INPUT_FILE}' tidak ditemukan!")
        return

    print(f"Total baris awal: {len(df)}")

    # 1. Ekstraksi Fitur Dasar
    df['tahun'] = df.apply(cari_tahun, axis=1)
    df['transmisi'] = df.apply(tentukan_transmisi, axis=1)
    df['model'] = df['judul'].apply(ekstrak_model)

    # 2. Pembersihan Angka
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    # 3. Filter Nilai Masuk Akal
    df = df.dropna(subset=['harga', 'tahun'])
    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 2_500_000_000)]
    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= 2026)]
    df['tahun'] = df['tahun'].astype(int)
    df['merek'] = df['merek'].astype(str).str.strip().str.title()

    # 4. Hapus Duplikasi
    if 'id_iklan' in df.columns:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(subset=['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh', 'harga'], keep='first')

    # 5. Imputasi Odometer
    median_global_km = df['jarak_tempuh'].dropna().median()
    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(median_global_km).astype(int)

    # ============================================================
    # 6. FEATURE ENGINEERING & GROUND TRUTH LIKUIDITAS
    # ============================================================
    print("Membentuk atribut rekayasa fitur & target likuiditas pasar...")

    # A. Kalkulasi Durasi Hari Iklan Tayang
    kolom_waktu = next((k for k in ['tanggal_posting', 'created_at', 'tanggal_iklan_dibuat'] if k in df.columns), None)
    if kolom_waktu:
        df['durasi_tayang_hari'] = df[kolom_waktu].apply(hitung_durasi_hari)
        median_durasi = df['durasi_tayang_hari'].dropna().median()
        if pd.isna(median_durasi):
            median_durasi = 14
        df['durasi_tayang_hari'] = df['durasi_tayang_hari'].fillna(median_durasi).astype(int)
    else:
        np.random.seed(42)
        df['durasi_tayang_hari'] = np.random.randint(1, 45, size=len(df))

    # B. Usia Mobil & Intensitas Pemakaian
    df['usia_mobil'] = 2026 - df['tahun']
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil'].apply(lambda x: max(1, x))).round(0).astype(int)

    # C. Deviasi Harga Pasar (Berdasarkan Merek, Model, dan Tahun)
    median_model_tahun = df.groupby(['model', 'tahun'])['harga'].transform('median')
    median_model = df.groupby('model')['harga'].transform('median')
    df['median_harga_pasar'] = median_model_tahun.fillna(median_model).fillna(df['harga'])
    df['deviasi_harga_pasar'] = ((df['harga'] - df['median_harga_pasar']) / df['median_harga_pasar'] * 100).round(2)

    # D. Pembentukan Label Target Likuiditas yang Realistis
    def tentukan_likuiditas_pasar(row):
        status_terjual = str(row.get('status_iklan_terjual', '')).lower() == 'terjual'
        durasi = row['durasi_tayang_hari']
        deviasi = row['deviasi_harga_pasar']
        km_tahun = row['km_per_tahun']

        # 1. Jika iklan sudah ditandai terjual oleh penjual
        if status_terjual:
            if durasi < 15:
                return 'Cepat'
            elif durasi <= 30:
                return 'Sedang'
            else:
                return 'Lambat'

        # 2. Jika iklan sudah tayang lama di marketplace dan belum terjual
        if durasi > 30:
            return 'Lambat'

        # 3. Jika iklan masih baru (1 - 30 hari) dan belum terjual
        # Evaluasi daya saing pasar unit (Kombinasi harga penawaran & kondisi pemakaian)
        if deviasi > 8.0 or km_tahun > 30000:
            return 'Lambat'  # Harga kemahalan atau km terlalu lelah -> pasar lambat menyerap
        elif deviasi <= -3.0 and km_tahun <= 18000:
            return 'Cepat'   # Harga di bawah pasar dan pemakaian wajar/rendah -> cepat laku
        else:
            return 'Sedang'  # Berada pada batas normal pasar

    df['kategori_penjualan'] = df.apply(tentukan_likuiditas_pasar, axis=1)

    # E. Deteksi Profil Penjual & Garansi dari Deskripsi (Text Mining)
    deskripsi_teks = df['deskripsi'].fillna('').str.lower()
    df['tipe_penjual'] = deskripsi_teks.apply(
        lambda t: 'Dealer' if any(k in t for k in ['showroom', 'paket kredit', 'tdp', 'dp ', 'otospector', 'olxmobbi', 'leasing', 'bca finance']) else 'Individu'
    )
    if 'tipe_penjual_badge' in df.columns:
        df.loc[df['tipe_penjual_badge'].astype(str).str.lower().isin(['pro', 'dealer', 'showroom']), 'tipe_penjual'] = 'Dealer'

    df['ada_garansi'] = deskripsi_teks.apply(
        lambda t: 'Ya' if any(k in t for k in ['garansi', 'warranty', 'sertifikat', 'otospector', 'lulus inspeksi', 'bebas banjir']) else 'Tidak'
    )

    # F. Normalisasi Teks
    df['deskripsi_bersih'] = df['deskripsi'].apply(bersihkan_dan_normalisasi_teks)
    df['judul_bersih'] = df['judul'].apply(bersihkan_dan_normalisasi_teks)

    df = df.reset_index(drop=True)

    # ============================================================
    # 7. SIMPAN HASIL PREPROCESSING
    # ============================================================
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[SUKSES] Dataset bersih tersimpan ke '{OUTPUT_FILE}' ({len(df)} baris)")
    print(f"Distribusi Target Likuiditas: {df['kategori_penjualan'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()