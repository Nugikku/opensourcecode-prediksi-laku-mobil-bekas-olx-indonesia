"""
AutoLaku - Script Preprocessing Data Mobil Bekas OLX
Versi 2: Diperkaya dengan laporan kualitas data menyeluruh (materi P3-4).

Perubahan utama dari versi sebelumnya:
  - Pencatatan jumlah baris di tiap tahap pipeline (bahan UTS nomor 2).
  - Laporan missing value per kolom, penanda baris yang diimputasi, dan
    perbandingan distribusi sebelum/sesudah imputasi.
  - Deteksi noise berbasis aturan domain (mis. km/tahun tidak masuk akal).
  - Deteksi outlier IQR pada log-harga per kelompok merek+usia; outlier
    DITANDAI (kolom is_outlier_harga), tidak langsung dibuang.
  - Smoothing TIDAK dipakai (data bukan deret waktu) — dicatat di laporan.
  - Data integration TIDAK relevan (sumber tunggal OLX) — dicatat di laporan.
  - Standardisasi merek, model, transmisi dengan ringkasan varian sebelum/sesudah.
  - Fitur teks diperluas: riwayat_negatif, catatan_servis, pajak_hidup,
    selain ada_garansi dan indikasi_kredit_tdp yang sudah ada.

Keluaran:
  - dataset_olx_bersih.csv  (dataset bersih untuk tahap selanjutnya)
  - laporan_preprocessing.txt  (laporan kualitas data)
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
LAPORAN_FILE = "laporan_preprocessing.txt"
TAHUN_SEKARANG = 2026

# Konstanta bersama — diimpor oleh modul lain supaya konsisten di satu tempat.
BRACKET_USIA_TAHUN = 5
AMBANG_KEYAKINAN_SAMPEL = 15
DEVIASI_MIN, DEVIASI_MAX = -90.0, 300.0

# ============================================================
# INFRASTRUKTUR LAPORAN
# ============================================================
_log_lines = []
_row_log = []


import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def _log(msg=""):
    """Cetak ke konsol dan simpan untuk laporan akhir."""
    _log_lines.append(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


def _catat_baris(tahap, jumlah, keterangan=""):
    """Catat jumlah baris di suatu tahap pipeline."""
    _row_log.append({"tahap": tahap, "jumlah_baris": jumlah, "keterangan": keterangan})
    ket = f" ({keterangan})" if keterangan else ""
    _log(f"  [{tahap}] -> {jumlah:,} baris{ket}")


# ============================================================
# 1. DAFTAR MODEL PER MEREK UNTUK EKSTRAKSI DARI JUDUL
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
    "jeep": ["rubicon", "wrangler", "sahara", "cherokee", "compass", "renegade"],
    "mini": ["cooper", "countryman", "clubman"],
    "hyundai": ["creta", "stargazer", "santa fe", "palisade", "ioniq", "tucson", "h-1"],
    "wuling": ["confero", "almaz", "cortez", "air ev", "binguo", "alvez", "formo"],
    "baic": ["bj40"],
    "byd": ["sealion"],
    "land rover": ["defender"],
    "lexus": ["rx300"],
    "ford": ["everest", "ranger"],
    "volkswagen": ["tiguan"],
    "kia": ["picanto", "rio", "soluto", "seltos", "sonet", "sportage", "carens", "grand carnival", "sorento"],
    "mazda": ["mazda2", "mazda3", "cx3", "cx5", "cx8", "cx9", "biante"],
    "chery": ["omoda 5", "omoda", "tiggo 8", "tiggo 7", "tiggo 5x", "tiggo cross", "tiggo", "j6t", "j6"],
    "chevrolet": ["trailblazer", "captiva", "colorado", "trax", "spin", "cruze", "aveo", "orlando"],
    "subaru": ["forester", "outback", "impreza", "brz", "wrx", "levorg", "xv"],
}

KAMUS_ALIAS_MODEL = {
    "inova": "innova", "innofa": "innova", "avanzaa": "avanza",
    "mazda 2": "mazda2", "mazda 3": "mazda3",
    "cx-3": "cx3", "cx 3": "cx3", "cx-5": "cx5", "cx 5": "cx5",
    "cx-8": "cx8", "cx 8": "cx8", "cx-9": "cx9", "cx 9": "cx9",
    "carnival": "grand carnival",
}

MODEL_PREFIX_AMBIGU = {
    "kijang": ["innova", "inova"],
}


def _buat_pola_regex(kata):
    dasar = r'\b' + re.escape(kata) + r'\b'
    kata_lebih_spesifik = MODEL_PREFIX_AMBIGU.get(kata)
    if kata_lebih_spesifik:
        pola_negatif = '|'.join(re.escape(k) for k in kata_lebih_spesifik)
        return dasar + rf'(?!\s*(?:{pola_negatif}))'
    return dasar


def ekstrak_model(row):
    model_asli = str(row.get('model', '')).strip()
    if model_asli and model_asli.lower() not in ['lainnya', 'others', 'nan', '']:
        return model_asli.title()

    merek_lower = str(row.get('merek', '')).strip().lower()
    judul_lower = str(row.get('judul', '')).lower()
    daftar_model_merek = MODEL_PER_MEREK.get(merek_lower, [])

    kandidat_kata = list(daftar_model_merek) + [
        alias for alias, baku in KAMUS_ALIAS_MODEL.items() if baku in daftar_model_merek
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
def is_harga_boneka(angka_str):
    """Deteksi harga placeholder/trik iklan: seluruh digit sama
    (mis. 99999999, 88888888) atau pola 2-digit berulang (mis. 12121212)."""
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


def deteksi_transmisi(row):
    """Return (transmisi, dari_teks). dari_teks=False berarti iklan TIDAK menyebut
    transmisi dan nilai 'automatic' hanyalah asumsi default."""
    gabungan = f"{row.get('transmisi', '')} {row.get('judul', '')} {str(row.get('deskripsi', ''))[:300]}".lower()
    if any(k in gabungan for k in [' matic', '-at', ' at ', ' a/t', 'otomatis', ' tiptronic', ' triptonic', ' cvt ']):
        return 'automatic', True
    elif any(k in gabungan for k in [' manual', '-mt', ' mt ', ' m/t']):
        return 'manual', True
    return 'automatic', False


def tentukan_transmisi(row):
    return deteksi_transmisi(row)[0]


def hitung_usia_bracket(usia_mobil, ukuran_bracket=BRACKET_USIA_TAHUN):
    """Kelompokkan usia mobil ke rentang N-tahunan (mis. 0-4, 5-9, 10-14)."""
    return (int(usia_mobil) // ukuran_bracket) * ukuran_bracket


# ============================================================
# 3b. EKSTRAKSI TEKS TERPADU (dipakai training & aplikasi)
# ============================================================
NEGASI = {"tidak", "bukan", "tanpa", "belum", "no", "nggak", "tdk", "ga", "gak"}
# Untuk riwayat negatif, "bebas" dan "free" berfungsi sbg negasi
# ("bebas banjir" = tidak ada riwayat banjir).
NEGASI_RIWAYAT = NEGASI | {"bebas", "free", "bersih"}

POLA_GARANSI = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|bebas tabrakan)\b'
POLA_RIWAYAT_NEGATIF = r'\b(banjir|tabrak|tabrakan|kecelakaan|bocor|keropos)\b'
POLA_CATATAN_SERVIS = r'\b(service record|catatan servis|buku servis|bengkel resmi|servis rutin|record servis)\b'
POLA_PAJAK_HIDUP = r'\b(pajak hidup|pajak panjang|pajak on|pjk hidup|pjk panjang|stnk hidup|stnk panjang)\b'
POLA_KREDIT = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit|kredit syariah)\b'


def _ada_pola_tanpa_negasi(teks, pola, negasi_set=None):
    """True jika pola ditemukan dan TIDAK didahului kata negasi (2 kata sebelumnya)."""
    if negasi_set is None:
        negasi_set = NEGASI
    for m in re.finditer(pola, teks):
        sebelum = teks[:m.start()].split()[-2:]
        if any(w in negasi_set for w in sebelum):
            continue
        return True
    return False


def ekstrak_fitur_teks(judul, deskripsi):
    """Fungsi terpadu: satu sumber kebenaran untuk training DAN prediksi.

    Mengembalikan dict dengan:
      judul_bersih, deskripsi_bersih, ada_garansi, riwayat_negatif,
      catatan_servis, pajak_hidup, indikasi_kredit_tdp
    """
    judul_b = bersihkan_dan_normalisasi_teks(judul)
    desk_b = bersihkan_dan_normalisasi_teks(deskripsi)
    gabung = f"{judul_b} {desk_b}"
    return {
        'judul_bersih': judul_b,
        'deskripsi_bersih': desk_b,
        'ada_garansi': 'Ya' if _ada_pola_tanpa_negasi(gabung, POLA_GARANSI) else 'Tidak',
        'riwayat_negatif': 'Ya' if _ada_pola_tanpa_negasi(gabung, POLA_RIWAYAT_NEGATIF, NEGASI_RIWAYAT) else 'Tidak',
        'catatan_servis': 'Ya' if _ada_pola_tanpa_negasi(gabung, POLA_CATATAN_SERVIS) else 'Tidak',
        'pajak_hidup': 'Ya' if _ada_pola_tanpa_negasi(gabung, POLA_PAJAK_HIDUP) else 'Tidak',
        'indikasi_kredit_tdp': 1 if _ada_pola_tanpa_negasi(gabung, POLA_KREDIT) else 0,
    }


def klip_deviasi(nilai):
    clipped = np.clip(nilai, DEVIASI_MIN, DEVIASI_MAX)
    if isinstance(nilai, (int, float, np.number)):
        return float(clipped)
    return clipped


# ============================================================
# 3c. HARGA WAJAR FALLBACK UNTUK KATEGORI BERSAMPEL SEDIKIT
# ============================================================
def hitung_harga_wajar_fallback(df_ref, merek, model, usia, exclude_idx=None):
    """Regresi log-harga ~ usia per model (min 5 data), lalu per merek (min 15).
    exclude_idx dipakai saat training (leave-one-out) agar tidak bocor.
    Return (harga, sumber) atau (None, None)."""
    d = df_ref if exclude_idx is None else df_ref.drop(index=exclude_idx, errors='ignore')
    for level, mask, minimal in (
        ("model", (d['merek'] == merek) & (d['model'] == model), 5),
        ("merek", d['merek'] == merek, 15),
    ):
        g = d[mask]
        if len(g) < minimal:
            continue
        if g['usia_mobil'].nunique() >= 3:
            b, a = np.polyfit(g['usia_mobil'], np.log(g['harga']), 1)
            if b < 0:
                pred = float(np.exp(a + b * usia))
                pred = float(np.clip(pred, g['harga'].min() * 0.6, g['harga'].max() * 1.4))
                return pred, f"regresi usia ({level})"
        return float(g['harga'].median()), f"median ({level})"
    return None, None


def pilih_harga_wajar(harga_model, sampel, df_ref, merek, model, usia, exclude_idx=None):
    """Sampel cukup -> pakai model ML. Sampel kurang -> pakai fallback bila tersedia."""
    if sampel >= AMBANG_KEYAKINAN_SAMPEL:
        return float(harga_model), "model ML"
    hf, sumber = hitung_harga_wajar_fallback(df_ref, merek, model, usia, exclude_idx)
    if hf is None:
        return float(harga_model), "model ML (data terbatas)"
    return hf, sumber


# ============================================================
# 4. DETEKSI NOISE & OUTLIER
# ============================================================
def deteksi_noise(df):
    """Deteksi noise berbasis aturan domain. Menambahkan kolom
    'is_noise' dan 'alasan_noise'. Baris TIDAK dihapus."""
    is_noise = pd.Series(False, index=df.index)
    alasan = pd.Series("", index=df.index)

    # Aturan 1: km/tahun > 80.000 → tidak masuk akal untuk mobil penumpang
    m1 = df['km_per_tahun'] > 80_000
    is_noise |= m1
    alasan = alasan.where(~m1, alasan + "km/tahun>80rb; ")

    # Aturan 2: km < 100 dan usia > 5 tahun → sangat mencurigakan
    m2 = (df['jarak_tempuh'] < 100) & (df['usia_mobil'] > 5)
    is_noise |= m2
    alasan = alasan.where(~m2, alasan + "km<100&usia>5; ")

    # Aturan 3: harga sangat rendah untuk merek premium (< Rp 30 juta)
    merek_premium = {'Bmw', 'Mercedes-Benz', 'Lexus', 'Land Rover', 'Jeep'}
    m3 = (df['merek'].isin(merek_premium)) & (df['harga'] < 30_000_000)
    is_noise |= m3
    alasan = alasan.where(~m3, alasan + "harga_premium_terlalu_rendah; ")

    df['is_noise'] = is_noise
    df['alasan_noise'] = alasan.str.rstrip('; ')
    return df


def deteksi_outlier_iqr(df):
    """Deteksi outlier dengan IQR pada log-harga per kelompok merek+usia_bracket.
    Outlier DITANDAI (kolom 'is_outlier_harga'), TIDAK dibuang — karena mobil
    mewah bisa merupakan data valid (pembeda noise vs outlier di P3-4).
    Kelompok dengan kurang dari 5 anggota dilewati (terlalu kecil utk IQR)."""
    df['is_outlier_harga'] = False
    df['log_harga'] = np.log(df['harga'])

    for (merek, bracket), idx in df.groupby(['merek', 'usia_bracket']).groups.items():
        if len(idx) < 5:
            continue
        grup = df.loc[idx, 'log_harga']
        q1, q3 = grup.quantile(0.25), grup.quantile(0.75)
        iqr = q3 - q1
        batas_bawah = q1 - 1.5 * iqr
        batas_atas = q3 + 1.5 * iqr
        outlier_mask = (grup < batas_bawah) | (grup > batas_atas)
        df.loc[idx[outlier_mask], 'is_outlier_harga'] = True

    df.drop(columns=['log_harga'], inplace=True)
    return df


# ============================================================
# 5. LAPORAN MISSING VALUE
# ============================================================
def laporan_missing_value(df, judul_tahap=""):
    """Cetak dan catat missing value per kolom."""
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    _log(f"\n--- Missing Value {judul_tahap} ---")
    if mv.empty:
        _log("  Tidak ada missing value.")
    else:
        total = len(df)
        for col, cnt in mv.items():
            _log(f"  {col:30s}: {cnt:5,} ({cnt/total*100:.1f}%)")
    return mv


# ============================================================
# 6. LAPORAN STANDARDISASI
# ============================================================
def laporan_standardisasi(sebelum, sesudah, nama_kolom):
    """Bandingkan varian unik sebelum dan sesudah standardisasi."""
    unik_sebelum = set(sebelum.dropna().unique())
    unik_sesudah = set(sesudah.dropna().unique())
    _log(f"\n--- Standardisasi '{nama_kolom}' ---")
    _log(f"  Varian sebelum: {len(unik_sebelum)} → sesudah: {len(unik_sesudah)}")
    hilang = unik_sebelum - unik_sesudah
    if hilang and len(hilang) <= 20:
        _log(f"  Varian yang dinormalisasi: {sorted(hilang)[:20]}")


# ============================================================
# 7. PIPELINE UTAMA
# ============================================================
def main():
    global _log_lines, _row_log
    _log_lines = []
    _row_log = []

    _log("=" * 75)
    _log("AUTOLAKU PREPROCESSING PIPELINE (P3-4)")
    _log("=" * 75)

    if not os.path.exists(INPUT_FILE):
        _log(f"[ERROR] File '{INPUT_FILE}' tidak ditemukan.")
        return

    df = pd.read_csv(INPUT_FILE)
    _catat_baris("01_data_mentah", len(df), "setelah pd.read_csv")

    # --- Laporan missing value awal ---
    laporan_missing_value(df, "(data mentah)")

    # -----------------------------------------------------------------
    # TAHAP 1: Ekstraksi Fitur Dasar
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 1: Ekstraksi fitur dasar (tahun, transmisi, model)")

    # Simpan varian asli untuk laporan standardisasi
    merek_asli = df['merek'].copy()
    transmisi_asli = df.get('transmisi', pd.Series(dtype=str)).copy()

    df['tahun'] = df.apply(cari_tahun, axis=1)
    hasil_trans = df.apply(deteksi_transmisi, axis=1)
    df['transmisi'] = hasil_trans.apply(lambda x: x[0])
    df['transmisi_dari_teks'] = hasil_trans.apply(lambda x: int(x[1]))
    n_asumsi = int((df['transmisi_dari_teks'] == 0).sum())
    _log(f"  Transmisi tidak disebut di iklan (diasumsikan 'automatic'): {n_asumsi:,} baris "
         f"({n_asumsi/len(df)*100:.1f}%) -> ditandai kolom transmisi_dari_teks=0")
    df['model'] = df.apply(ekstrak_model, axis=1)

    _catat_baris("02_setelah_ekstraksi", len(df), "belum ada baris yang dihapus")

    # -----------------------------------------------------------------
    # TAHAP 2: Pembersihan Angka
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 2: Pembersihan angka (harga, jarak_tempuh)")
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    n_harga_null = int(df['harga'].isnull().sum())
    n_km_null = int(df['jarak_tempuh'].isnull().sum())
    _log(f"  Harga tidak valid/boneka → NULL: {n_harga_null:,}")
    _log(f"  Jarak tempuh tidak valid → NULL: {n_km_null:,}")

    # -----------------------------------------------------------------
    # TAHAP 3: Filter Validitas Harga & Tahun
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 3: Filter validitas harga & tahun")
    df = df.dropna(subset=['harga', 'tahun'])
    _catat_baris("03_setelah_dropna_harga_tahun", len(df), "drop baris tanpa harga/tahun")

    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 3_000_000_000)]
    _catat_baris("04_setelah_filter_harga", len(df), "harga 25jt-3M")

    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= TAHUN_SEKARANG)]
    df['tahun'] = df['tahun'].astype(int)
    _catat_baris("05_setelah_filter_tahun", len(df), f"tahun 1995-{TAHUN_SEKARANG}")

    # -----------------------------------------------------------------
    # TAHAP 4: Standardisasi Format
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 4: Standardisasi format (merek, model, transmisi)")
    df['merek'] = df['merek'].astype(str).str.strip().str.title()
    df = df[~df['merek'].str.lower().isin(['lainnya', 'others', 'nan', ''])]
    _catat_baris("06_setelah_filter_merek", len(df), "hapus merek tidak valid")

    # Laporan standardisasi
    laporan_standardisasi(merek_asli, df['merek'], 'merek')
    laporan_standardisasi(transmisi_asli, df['transmisi'], 'transmisi')

    _log(f"\n  Distribusi transmisi: {dict(df['transmisi'].value_counts())}")
    _log(f"  Merek unik: {df['merek'].nunique()}")
    _log(f"  Model unik: {df['model'].nunique()}")

    # -----------------------------------------------------------------
    # TAHAP 5: Imputasi Jarak Tempuh (dengan tracking)
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 5: Imputasi jarak tempuh")
    km_null_mask = df['jarak_tempuh'].isnull()
    n_diimputasi = int(km_null_mask.sum())
    pct_diimputasi = n_diimputasi / len(df) * 100
    _log(f"  Baris yang perlu diimputasi: {n_diimputasi:,} ({pct_diimputasi:.1f}%)")

    # Distribusi SEBELUM imputasi (hanya yang tidak null)
    km_valid = df.loc[~km_null_mask, 'jarak_tempuh']
    dist_sebelum = {
        'mean': float(km_valid.mean()) if len(km_valid) else 0,
        'median': float(km_valid.median()) if len(km_valid) else 0,
        'std': float(km_valid.std()) if len(km_valid) else 0,
        'min': float(km_valid.min()) if len(km_valid) else 0,
        'max': float(km_valid.max()) if len(km_valid) else 0,
    }

    # Tandai baris yang diimputasi
    df['jarak_tempuh_diimputasi'] = km_null_mask

    # Imputasi: median per tahun, lalu fallback median global
    median_global = km_valid.median() if len(km_valid) else 50000
    if pd.isna(median_global):
        median_global = 50000
    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(
        lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(median_global).astype(int)

    # Distribusi SESUDAH imputasi
    dist_sesudah = {
        'mean': float(df['jarak_tempuh'].mean()),
        'median': float(df['jarak_tempuh'].median()),
        'std': float(df['jarak_tempuh'].std()),
        'min': float(df['jarak_tempuh'].min()),
        'max': float(df['jarak_tempuh'].max()),
    }

    _log(f"\n  Perbandingan distribusi jarak_tempuh:")
    _log(f"  {'Statistik':12s} | {'Sebelum':>15s} | {'Sesudah':>15s}")
    _log(f"  {'-'*12}-+-{'-'*15}-+-{'-'*15}")
    for stat in ['mean', 'median', 'std', 'min', 'max']:
        _log(f"  {stat:12s} | {dist_sebelum[stat]:>15,.0f} | {dist_sesudah[stat]:>15,.0f}")

    if pct_diimputasi > 5:
        _log(f"\n  [PERHATIAN] {pct_diimputasi:.1f}% baris diimputasi (>5%). "
             "Imputasi median per tahun dihitung dari seluruh data (termasuk test). "
             "Jika ini menjadi masalah, imputasi harus dipindahkan ke dalam pipeline CV.")

    # -----------------------------------------------------------------
    # TAHAP 6: Feature Engineering
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 6: Feature engineering")
    df['usia_mobil'] = df['tahun'].apply(lambda x: max(1, int(TAHUN_SEKARANG - x)))
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil']).round().astype(int)

    # -----------------------------------------------------------------
    # TAHAP 7: Deteksi Noise (aturan domain)
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 7: Deteksi noise berbasis aturan domain")
    df = deteksi_noise(df)
    n_noise = int(df['is_noise'].sum())
    _log(f"  Baris ditandai noise: {n_noise:,} ({n_noise/len(df)*100:.1f}%)")
    if n_noise > 0:
        alasan_counts = df.loc[df['is_noise'], 'alasan_noise'].value_counts()
        for alasan, cnt in alasan_counts.items():
            _log(f"    - {alasan}: {cnt:,}")
    _log("  [CATATAN] Baris noise DITANDAI, TIDAK dihapus.")

    # -----------------------------------------------------------------
    # TAHAP 8: Deteksi Outlier (IQR pada log-harga)
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 8: Deteksi outlier IQR pada log-harga per merek+usia_bracket")
    df['usia_bracket'] = df['usia_mobil'].apply(hitung_usia_bracket)
    df = deteksi_outlier_iqr(df)
    n_outlier = int(df['is_outlier_harga'].sum())
    _log(f"  Baris ditandai outlier harga: {n_outlier:,} ({n_outlier/len(df)*100:.1f}%)")
    _log("  [CATATAN] Outlier DITANDAI, TIDAK dihapus — mobil mewah bisa valid.")
    _log("  [CATATAN P3-4] Pembeda noise vs outlier: noise = data SALAH (entri error), "
         "outlier = data BENAR tapi EKSTREM (mobil langka/mewah).")

    # -----------------------------------------------------------------
    # TAHAP 9: Ekstraksi Fitur Teks (diperluas)
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 9: Ekstraksi fitur teks (diperluas)")
    fitur_teks = df.apply(
        lambda r: ekstrak_fitur_teks(r.get('judul'), r.get('deskripsi')),
        axis=1, result_type='expand')
    for kolom in fitur_teks.columns:
        df[kolom] = fitur_teks[kolom]

    for kolom_teks in ['ada_garansi', 'riwayat_negatif', 'catatan_servis', 'pajak_hidup']:
        vc = df[kolom_teks].value_counts()
        _log(f"  {kolom_teks}: {dict(vc)}")
    _log(f"  indikasi_kredit_tdp: {dict(df['indikasi_kredit_tdp'].value_counts())}")

    # -----------------------------------------------------------------
    # TAHAP 10: Normalisasi Tipe Penjual
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 10: Normalisasi tipe penjual")
    if 'tipe_penjual_badge' in df.columns:
        df['tipe_penjual'] = df['tipe_penjual_badge'].apply(
            lambda x: 'Dealer' if pd.notna(x) and str(x).strip().lower()
            not in ['individu', 'none', '', 'nan'] else 'Individu')
    else:
        df['tipe_penjual'] = 'Individu'
    _log(f"  Distribusi: {dict(df['tipe_penjual'].value_counts())}")

    # -----------------------------------------------------------------
    # TAHAP 11: Hapus Duplikat
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 11: Hapus duplikat")
    sebelum_dedup = len(df)
    if 'id_iklan' in df.columns and df['id_iklan'].notna().sum() > 0:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(
        subset=['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh', 'harga'], keep='first')
    df = df.reset_index(drop=True)
    n_duplikat = sebelum_dedup - len(df)
    _catat_baris("07_setelah_dedup", len(df), f"hapus {n_duplikat:,} duplikat")

    # -----------------------------------------------------------------
    # TAHAP 12: Indikator Kecukupan Sampel
    # -----------------------------------------------------------------
    _log("\n>>> TAHAP 12: Indikator kecukupan sampel")
    df['jumlah_sampel_kategori'] = df.groupby(
        ['merek', 'model', 'usia_bracket'])['model'].transform('count')
    n_rendah = int((df['jumlah_sampel_kategori'] < AMBANG_KEYAKINAN_SAMPEL).sum())
    _log(f"  {n_rendah:,} dari {len(df):,} baris ({n_rendah/len(df)*100:.1f}%) berada di kombinasi "
         f"merek+model+usia dgn sampel < {AMBANG_KEYAKINAN_SAMPEL}")

    # -----------------------------------------------------------------
    # CATATAN METODOLOGI (P3-4)
    # -----------------------------------------------------------------
    _log("\n" + "=" * 75)
    _log("CATATAN METODOLOGI (P3-4)")
    _log("=" * 75)
    _log("\n[SMOOTHING] Tidak dipakai. Teknik smoothing (moving average, exponential "
         "smoothing, dll.) cocok untuk data deret waktu (time series). Data ini "
         "adalah data cross-sectional (snapshot iklan), bukan urutan temporal, "
         "sehingga smoothing tidak relevan.")
    _log("\n[DATA INTEGRATION] Tidak relevan. Data bersumber tunggal dari OLX. "
         "Tidak ada penggabungan dari sumber lain yang memerlukan resolusi "
         "entitas, penanganan konflik, atau deduplikasi lintas-sumber.")

    # -----------------------------------------------------------------
    # RINGKASAN AKHIR
    # -----------------------------------------------------------------
    _log("\n" + "=" * 75)
    _log("RINGKASAN PIPELINE")
    _log("=" * 75)
    _log(f"\nData bersih siap tahap selanjutnya: {len(df):,} baris")
    _log(f"Rentang Harga : Rp {int(df['harga'].min()):,} s/d Rp {int(df['harga'].max()):,}")
    _log(f"Merek Teratas :\n{df['merek'].value_counts().head(5).to_string()}")

    _log("\nAlur baris:")
    for entry in _row_log:
        _log(f"  {entry['tahap']:40s} {entry['jumlah_baris']:>6,}  {entry['keterangan']}")

    # --- Simpan dataset ---
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    _log(f"\n[SUKSES] Dataset berhasil disimpan ke '{OUTPUT_FILE}'.")

    # --- Simpan laporan preprocessing ---
    with open(LAPORAN_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_log_lines))
    _log(f"[SUKSES] Laporan preprocessing disimpan ke '{LAPORAN_FILE}'.")


if __name__ == "__main__":
    main()