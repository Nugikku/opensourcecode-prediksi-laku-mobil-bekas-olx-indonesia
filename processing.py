"""
AutoLaku - Script Preprocessing Data Mobil Bekas OLX
Menggabungkan keandalan ekstraksi script lama dengan fitur analitis DSS.

Riwayat revisi (lihat komentar berkode [FIX] / [BARU] di bawah):
  [FIX-1] Ekstraksi model dari judul dibatasi ke daftar model milik MEREK
          baris itu sendiri (bukan mencari ke seluruh 170+ model lintas
          merek). Sebelumnya judul yang menyebut merek lain utk SEO/
          perbandingan (mis. "TT fortuner", "BUKAN CAMRY") bisa salah
          ke-label sebagai merek yang disebut, bukan merek aslinya.
  [FIX-2] Saat beberapa model dari merek yang sama muncul di judul, yang
          dipilih adalah yang MUNCUL PALING AWAL di teks, dan varian
          ejaan/typo umum (mis. "inova" utk "innova") dikenali lewat
          KAMUS_ALIAS_MODEL.
  [FIX-2b] Pengecualian utk nama model yang jadi PREFIX model lain yang
          lebih spesifik & beda kelas harga jauh (mis. "Kijang" klasik
          vs "Kijang Innova" modern -- nama resmi Innova di Indonesia
          tetap menyertakan kata "Kijang").
  [FIX-3] Harga "boneka"/placeholder (mis. 99999999, 88888888) yang
          sering dipakai penjual OLX sbg trik iklan disaring -- dulu
          lolos filter rentang harga & mencemari data training.
  [DICABUT] Sempat dicoba mengonsolidasikan kombinasi merek+model bersampel
          sedikit ke satu kategori umum "Model Lainnya" per merek. Setelah
          diuji, langkah ini JUSTRU MEMPERBURUK akurasi (mencampur unit
          murah & mahal yang tidak sejenis dalam satu kategori generik),
          sehingga dihapus lagi.
  [BARU] Ditambahkan kolom 'usia_bracket' & 'jumlah_sampel_kategori':
          bukan mengubah data atau menyembunyikan kelangkaan sampel,
          tapi MENGUKUR & MENYIMPAN informasi kelangkaannya, supaya
          tahap training & aplikasi prediksi bisa memberi peringatan
          keyakinan rendah ke pengguna alih-alih diam-diam salah.
  [BARU-2] ekstrak_fitur_teks(): SATU fungsi pembaca judul/deskripsi yang
          dipakai training DAN aplikasi (dulu dua versi berbeda), kini
          mengenali negasi ("tidak garansi", "tanpa garansi" = Tidak).
  [BARU-3] hitung_harga_wajar_fallback() & pilih_harga_wajar(): utk
          kombinasi bersampel sedikit, harga wajar dihitung dgn regresi
          log-harga terhadap usia (per model, lalu per merek). Pada uji
          leave-one-out galatnya lebih kecil daripada model ML utk kasus ini.
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

# [BARU] Konstanta bersama untuk indikator keyakinan prediksi. Diimpor
# langsung oleh train_model_klasifikasi.py & test_prediksi_klasifikasi.py
# supaya ambang & ukuran kelompok usia konsisten di satu tempat saja.
BRACKET_USIA_TAHUN = 5      # ukuran kelompok usia (tahun) utk menghitung kecukupan sampel
AMBANG_KEYAKINAN_SAMPEL = 15  # di bawah ini, prediksi dianggap keyakinan RENDAH

# ============================================================
# 1. DAFTAR MODEL PER MEREK UNTUK EKSTRAKSI DARI JUDUL [FIX-1]
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

# [FIX-2] Alias/typo umum -> nama model baku. Dicek untuk merek yang relevan.
KAMUS_ALIAS_MODEL = {
    "inova": "innova",
    "innofa": "innova",
    "avanzaa": "avanza",
    "mazda 2": "mazda2",
    "mazda 3": "mazda3",
    "cx-3": "cx3", "cx 3": "cx3",
    "cx-5": "cx5", "cx 5": "cx5",
    "cx-8": "cx8", "cx 8": "cx8",
    "cx-9": "cx9", "cx 9": "cx9",
    "carnival": "grand carnival",
}

# [FIX-2b] Beberapa nama model adalah PREFIX dari model lain yang lebih spesifik
# dan bernilai jual jauh berbeda. Kalau kata kunci yang lebih spesifik (value)
# ikut disebut, kata kunci generik (key) TIDAK boleh dianggap match sendiri.
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
        # [FIX-2] pilih yang muncul PALING AWAL di judul, bukan urutan daftar
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
    """[FIX-3] Deteksi harga placeholder/trik iklan: seluruh digit sama
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
    if is_harga_boneka(angka_saja):  # [FIX-3]
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
    if any(k in gabungan for k in [' matic', '-at', ' at ', ' a/t', 'otomatis', ' tiptronic', ' triptonic', ' cvt ']):
        return 'automatic'
    elif any(k in gabungan for k in [' manual', '-mt', ' mt ', ' m/t']):
        return 'manual'
    return 'automatic'

def hitung_usia_bracket(usia_mobil, ukuran_bracket=BRACKET_USIA_TAHUN):
    """[BARU] Kelompokkan usia mobil ke rentang N-tahunan (mis. 0-4, 5-9,
    10-14, ...) supaya perhitungan kecukupan sampel tidak terlalu ketat
    per-tahun-persis (yang hampir pasti selalu sedikit)."""
    return (int(usia_mobil) // ukuran_bracket) * ukuran_bracket


# ============================================================
# 3b. EKSTRAKSI TEKS TERPADU (dipakai training & aplikasi) [BARU-2]
# ============================================================
NEGASI = {"tidak", "bukan", "tanpa", "belum", "no", "nggak"}
POLA_GARANSI = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|service record|buku servis|catatan servis)\b'
POLA_KREDIT = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit|kredit syariah)\b'
DEVIASI_MIN, DEVIASI_MAX = -90.0, 300.0  # batas wajar deviasi harga (persen)

def _ada_pola_tanpa_negasi(teks, pola):
    """True jika pola ditemukan dan TIDAK didahului kata negasi (2 kata sebelumnya)."""
    for m in re.finditer(pola, teks):
        sebelum = teks[:m.start()].split()[-2:]
        if any(w in NEGASI for w in sebelum):
            continue
        return True
    return False

def ekstrak_fitur_teks(judul, deskripsi):
    judul_b = bersihkan_dan_normalisasi_teks(judul)
    desk_b = bersihkan_dan_normalisasi_teks(deskripsi)
    gabung = f"{judul_b} {desk_b}"
    return {
        'judul_bersih': judul_b,
        'deskripsi_bersih': desk_b,
        'ada_garansi': 'Ya' if _ada_pola_tanpa_negasi(gabung, POLA_GARANSI) else 'Tidak',
        'indikasi_kredit_tdp': 1 if _ada_pola_tanpa_negasi(gabung, POLA_KREDIT) else 0,
    }

def klip_deviasi(nilai):
    return float(np.clip(nilai, DEVIASI_MIN, DEVIASI_MAX))

# ============================================================
# 3c. HARGA WAJAR FALLBACK UNTUK KATEGORI BERSAMPEL SEDIKIT [BARU-3]
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

    # 7. Ekstraksi Fitur Teks (Garansi & Indikasi TDP Kredit) - fungsi terpadu [BARU-2]
    fitur_teks = df.apply(lambda r: ekstrak_fitur_teks(r.get('judul'), r.get('deskripsi')),
                          axis=1, result_type='expand')
    for kolom in fitur_teks.columns:
        df[kolom] = fitur_teks[kolom]

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

    # 10. [BARU] Indikator kecukupan sampel per merek+model+kelompok usia
    df['usia_bracket'] = df['usia_mobil'].apply(hitung_usia_bracket)
    df['jumlah_sampel_kategori'] = df.groupby(['merek', 'model', 'usia_bracket'])['model'].transform('count')

    n_rendah = int((df['jumlah_sampel_kategori'] < AMBANG_KEYAKINAN_SAMPEL).sum())
    print(f"[INFO] {n_rendah:,} dari {len(df):,} baris ({n_rendah/len(df)*100:.1f}%) berada di kombinasi "
          f"merek+model+usia dgn sampel < {AMBANG_KEYAKINAN_SAMPEL} (akan ditandai keyakinan rendah saat prediksi)")

    print(f"Data bersih siap training: {len(df):,} baris")
    print(f"Rentang Harga : Rp {int(df['harga'].min()):,} s/d Rp {int(df['harga'].max()):,}")
    print(f"Merek Teratas :\n{df['merek'].value_counts().head(5)}")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[SUKSES] Dataset berhasil disimpan ke '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()