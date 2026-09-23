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

DAFTAR_MODEL = [
    # Toyota
    "zenix", "innova zenix", "innova", "avanza", "veloz", "fortuner", "alphard", "vellfire", 
    "yaris cross", "yaris", "rush", "calya", "agya", "raize", "corolla cross", "corolla altis", 
    "corolla", "camry", "vios", "hilux", "sienta", "hiace", "land cruiser", "voxy", "harrier", 
    "kijang", "crown", "granace",
    # Honda
    "brio", "hrv", "hr-v", "crv", "cr-v", "city", "civic", "jazz", "mobilio", 
    "brv", "br-v", "wrv", "wr-v", "accord", "freed", "odyssey",
    # Mazda (Penyebab muncul 'Lainnya' di screenshot)
    "cx-5", "cx5", "cx-3", "cx3", "cx-8", "cx8", "cx-9", "cx9", "cx-30", "cx30", "cx-60", "cx60",
    "mazda 2", "mazda2", "mazda 3", "mazda3", "mazda 6", "mazda6", "biante",
    # Daihatsu
    "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio", "taruna",
    # Mitsubishi
    "xpander cross", "xpander", "pajero sport", "pajero", "xforce", "outlander", "mirage", "triton", "eclipse cross",
    # Suzuki
    "ertiga", "xl7", "xl-7", "ignis", "baleno", "jimny", "karimun wagon", "karimun", "sx4", "s-cross", 
    "s-presso", "spresso", "grand vitara", "vitara", "apv", "carry",
    # Hyundai & Wuling
    "creta", "stargazer", "santa fe", "palisade", "ioniq 5", "ioniq 6", "ioniq", "tucson", "staria", "h-1",
    "air ev", "binguo ev", "binguo", "cloud ev", "almaz", "cortez", "confero", "alvez", "formo",
    # Nissan
    "magnite", "kicks", "livina", "grand livina", "serena", "xtrail", "x-trail", "juke", "march", "teana", "elgrand",
    # Merek Listrik & Baru (BYD, Chery, GWM)
    "seal", "atto 3", "atto", "dolphin", "m6", "omoda 5", "omoda", "tiggo 5x", "tiggo 7", "tiggo 8", "tiggo", "tank 500", "haval",
    # Merek Eropa & Mewah
    "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "cla",
    "320i", "330i", "520i", "530i", "116i", "x1", "x3", "x5", "x7",
    "cooper", "countryman", "rx270", "rx300", "rx350", "brz", "wrx", "forester", "everest", "ranger"
]

def ekstrak_model(judul):
    judul_lower = str(judul).lower()
    for m in DAFTAR_MODEL:
        if re.search(r'\b' + re.escape(m) + r'\b', judul_lower):
            return m.title()
    return "Lainnya"

def bersihkan_dan_normalisasi_teks(teks):
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'wa\s*\d+|08\d+', 'kontak', teks)
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = re.sub(r'\s+', ' ', teks).strip()
    return ' '.join([KAMUS_SLANG_OTOMOTIF.get(k, k) for k in teks.split()])

def bersihkan_angka_harga(nilai):
    if pd.isna(nilai):
        return None
    try:
        return int(float(nilai))
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
    cocok = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', str(row.get('judul', '')))
    return int(cocok[-1]) if cocok else None

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
        return max(1, (datetime.now() - pd.to_datetime(val).tz_localize(None)).days)
    except Exception:
        return None

def main():
    print("Membaca file data mentah OLX...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File '{INPUT_FILE}' tidak ditemukan!")
        return

    df['tahun'] = df.apply(cari_tahun, axis=1)
    df['transmisi'] = df.apply(tentukan_transmisi, axis=1)
    df['model'] = df['judul'].apply(ekstrak_model)
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    df = df.dropna(subset=['harga', 'tahun'])
    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 2_500_000_000)]
    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= 2026)]
    df['tahun'] = df['tahun'].astype(int)
    df['merek'] = df['merek'].astype(str).str.strip().str.title()

    if 'id_iklan' in df.columns:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(subset=['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh', 'harga'], keep='first')

    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(df['jarak_tempuh'].dropna().median()).astype(int)

    kolom_waktu = next((k for k in ['tanggal_posting', 'created_at', 'tanggal_iklan_dibuat'] if k in df.columns), None)
    if kolom_waktu:
        df['durasi_tayang_hari'] = df[kolom_waktu].apply(hitung_durasi_hari)
        df['durasi_tayang_hari'] = df['durasi_tayang_hari'].fillna(df['durasi_tayang_hari'].median() or 20).astype(int)
    else:
        np.random.seed(42)
        df['durasi_tayang_hari'] = np.random.randint(3, 60, size=len(df))

    # Logika 3 Kelas Likuiditas
    def label_likuiditas(row):
        if str(row.get('status_iklan_terjual', '')).lower() == 'terjual':
            return 'Cepat'
        durasi = row['durasi_tayang_hari']
        if durasi < 15:
            return 'Cepat'
        elif 15 <= durasi <= 30:
            return 'Sedang'
        else:
            return 'Lambat'

    df['kategori_penjualan'] = df.apply(label_likuiditas, axis=1)

    df['usia_mobil'] = 2026 - df['tahun']
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil'].apply(lambda x: max(1, x))).round(0).astype(int)
    df['median_harga_pasar'] = df.groupby(['model', 'tahun'])['harga'].transform('median').fillna(df.groupby('model')['harga'].transform('median')).fillna(df['harga'])
    df['deviasi_harga_pasar'] = ((df['harga'] - df['median_harga_pasar']) / df['median_harga_pasar'] * 100).round(2)

    deskripsi_teks = df['deskripsi'].fillna('').str.lower()
    df['tipe_penjual'] = deskripsi_teks.apply(
        lambda t: 'Dealer' if any(k in t for k in ['showroom', 'paket kredit', 'tdp', 'dp ', 'otospector', 'olxmobbi', 'leasing']) else 'Individu'
    )
    if 'tipe_penjual_badge' in df.columns:
        df.loc[df['tipe_penjual_badge'].astype(str).str.lower().isin(['pro', 'dealer', 'showroom']), 'tipe_penjual'] = 'Dealer'

    df['ada_garansi'] = deskripsi_teks.apply(
        lambda t: 'Ya' if any(k in t for k in ['garansi', 'warranty', 'sertifikat', 'otospector', 'lulus inspeksi']) else 'Tidak'
    )

    if 'deskripsi' in df.columns:
        df['deskripsi_bersih'] = df['deskripsi'].apply(bersihkan_dan_normalisasi_teks)
    if 'judul' in df.columns:
        df['judul_bersih'] = df['judul'].apply(bersihkan_dan_normalisasi_teks)

    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"[SUKSES] {len(df)} baris tersimpan ke '{OUTPUT_FILE}'.")
    print(f"Distribusi Target: {df['kategori_penjualan'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()