"""
AutoLiquid DSS: Prediksi Likuiditas Mobil Bekas (Cepat/Sedang/Lambat)
Arsitektur: Classifier ML (dilatih dari label proxy) + info harga wajar
Fokus: Input Mandiri Penjual

[BARU] Menampilkan peringatan keyakinan rendah kalau kombinasi
merek+model+usia yang dipilih penjual punya sampel pembanding terlalu
sedikit di dataset (lihat AMBANG_KEYAKINAN_SAMPEL di processing.py).
"""

import os
import re
import pickle
import pandas as pd

try:
    from processing import AMBANG_KEYAKINAN_SAMPEL, BRACKET_USIA_TAHUN, hitung_usia_bracket
except ImportError:
    AMBANG_KEYAKINAN_SAMPEL = 15
    BRACKET_USIA_TAHUN = 5
    def hitung_usia_bracket(usia_mobil, ukuran_bracket=BRACKET_USIA_TAHUN):
        return (int(usia_mobil) // ukuran_bracket) * ukuran_bracket

FILE_MODEL_KLASIFIKASI = "model_klasifikasi_likuiditas.pkl"
FILE_MODEL_HARGA = "model_regresi_harga.pkl"
FILE_DATA = "dataset_olx_bersih.csv"
TAHUN_SEKARANG = 2026

for f_check in (FILE_MODEL_KLASIFIKASI, FILE_MODEL_HARGA, FILE_DATA):
    if not os.path.exists(f_check):
        print(f"[!] File '{f_check}' tidak ditemukan. Jalankan train_model_klasifikasi.py terlebih dahulu.")
        exit()

with open(FILE_MODEL_KLASIFIKASI, "rb") as f:
    model_klasifikasi = pickle.load(f)

with open(FILE_MODEL_HARGA, "rb") as f:
    model_harga = pickle.load(f)

df = pd.read_csv(FILE_DATA)

# Data referensi: hanya dipakai untuk menyusun daftar merek/model yang valid
# (agar input penjual konsisten dengan kategori yang dikenal model), dan
# untuk menghitung kecukupan sampel pembanding [BARU].
df_valid = df[
    (df['model'].fillna('').str.lower() != 'lainnya') &
    (df['merek'].fillna('').str.lower() != 'lainnya') &
    (df['model'].fillna('').str.strip() != '') &
    (df['harga'].notna()) &
    (df['tahun'].notna()) &
    (df['transmisi'].notna())
].copy().reset_index(drop=True)

print("=" * 80)
print("     AUTOLIQUID DSS: PREDIKSI LIKUIDITAS MOBIL BEKAS (PENJUAL)     ")
print("=" * 80)

# ============================================================
# 1. SELEKSI MEREK & MODEL (dari kategori yang dikenal model)
# ============================================================
daftar_merek = sorted(df_valid['merek'].unique())
print("\nPILIH MEREK KENDARAAN:")
for i in range(0, len(daftar_merek), 4):
    baris_teks = ""
    for j in range(4):
        if i + j < len(daftar_merek):
            baris_teks += f"[{i + j + 1:<2}] {daftar_merek[i + j]:<16} "
    print(baris_teks)

while True:
    try:
        no_m = int(input(f"\nPilih nomor merek (1-{len(daftar_merek)}): ").strip())
        if 1 <= no_m <= len(daftar_merek):
            merek_terpilih = daftar_merek[no_m - 1]
            break
        print("[!] Nomor tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka valid.")

df_merek = df_valid[df_valid['merek'] == merek_terpilih].copy()
daftar_model = sorted(df_merek['model'].unique())
print(f"\nPilihan Model untuk {merek_terpilih}:")
for i in range(0, len(daftar_model), 4):
    baris_teks = ""
    for j in range(4):
        if i + j < len(daftar_model):
            baris_teks += f"[{i + j + 1:<2}] {daftar_model[i + j]:<16} "
    print(baris_teks)

while True:
    try:
        no_mod = int(input(f"\nPilih nomor model (1-{len(daftar_model)}): ").strip())
        if 1 <= no_mod <= len(daftar_model):
            model_terpilih = daftar_model[no_mod - 1]
            break
        print("[!] Nomor tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka valid.")

# ============================================================
# 2. FORMULIR INPUT MANDIRI PENJUAL
# ============================================================
print("\n" + "=" * 80)
print("FORMULIR INPUT MANDIRI PENJUAL")
print("=" * 80)
tahun_unit = int(input(f"Masukkan Tahun Perakitan Mobil (1995-{TAHUN_SEKARANG}): ").strip())
transmisi_unit = input("Transmisi (manual/automatic): ").strip().lower()
km_unit = int(input("Masukkan Total Kilometer (KM): ").strip())
harga_iklan = float(input("Masukkan Rencana Harga Jual (Rp): ").strip())
tipe_penjual = "Individu"

print("\nKetik Rencana Deskripsi Iklan:")
deskripsi_teks = input("> ").strip().lower()

pola_garansi = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|service record|buku servis|terawat)\b'
pola_kredit = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit)\b'
ada_garansi = 'Ya' if re.search(pola_garansi, deskripsi_teks) else 'Tidak'
indikasi_tdp = 1 if re.search(pola_kredit, deskripsi_teks) else 0

# ============================================================
# 3. PREDIKSI: KLASIFIKASI LIKUIDITAS (utama) + HARGA WAJAR (info)
# ============================================================
usia_mobil = max(1, TAHUN_SEKARANG - tahun_unit)
km_per_tahun = round(km_unit / usia_mobil)
usia_bracket_input = hitung_usia_bracket(usia_mobil)

data_uji = pd.DataFrame([{
    'merek': merek_terpilih,
    'model': model_terpilih,
    'transmisi': transmisi_unit,
    'tipe_penjual': tipe_penjual,
    'ada_garansi': ada_garansi,
    'tahun': tahun_unit,
    'jarak_tempuh': km_unit,
    'usia_mobil': usia_mobil,
    'km_per_tahun': km_per_tahun
}])

# Vonis utama: langsung dari classifier terlatih (bukan aturan manual)
status_likuiditas = model_klasifikasi.predict(data_uji)[0]

proba_dict = None
if hasattr(model_klasifikasi.named_steps['clf'], "predict_proba"):
    kelas = model_klasifikasi.named_steps['clf'].classes_
    proba = model_klasifikasi.predict_proba(data_uji)[0]
    proba_dict = dict(zip(kelas, proba))

# Info pendukung: estimasi harga wajar (bukan penentu vonis, hanya konteks)
harga_wajar_ml = float(model_harga.predict(data_uji)[0])
deviasi_persen = ((harga_iklan - harga_wajar_ml) / harga_wajar_ml) * 100

# [BARU] Cek kecukupan sampel pembanding utk kombinasi merek+model+usia ini
jumlah_sampel_pembanding = int((
    (df['merek'] == merek_terpilih) &
    (df['model'] == model_terpilih) &
    (df['usia_mobil'].apply(hitung_usia_bracket) == usia_bracket_input)
).sum())
keyakinan_rendah = jumlah_sampel_pembanding < AMBANG_KEYAKINAN_SAMPEL

# ============================================================
# 4. OUTPUT KEPUTUSAN
# ============================================================
print("\n" + "=" * 80)
print("                   AUTOLIQUID DSS: KEPUTUSAN PASAR                    ")
print("=" * 80)
print(f"Spesifikasi        : {merek_terpilih} {model_terpilih} ({tahun_unit}) | {transmisi_unit.title()}")
print(f"Odometer           : {km_unit:,} KM ({km_per_tahun:,} KM/Tahun)")
print(f"Deteksi Garansi    : {ada_garansi}")
print(f"Indikasi TDP/Kredit: {'Ya' if indikasi_tdp else 'Tidak'}")
print("-" * 80)
print(f"Harga Penawaran    : Rp {int(harga_iklan):,}")
print(f"Estimasi Harga Wajar (info): Rp {int(harga_wajar_ml):,}  (deviasi {deviasi_persen:+.1f}%)")
print("-" * 80)
print(f"VONIS LIKUIDITAS   : [{status_likuiditas.upper()} LAKU]")
if proba_dict:
    print("Keyakinan model    : " + " | ".join(f"{k}: {v*100:.1f}%" for k, v in proba_dict.items()))
print(f"Sampel Pembanding  : {jumlah_sampel_pembanding} unit sejenis (merek+model+rentang usia serupa) di dataset")
if keyakinan_rendah:
    print("-" * 80)
    print(f"[!] PERINGATAN: Sampel pembanding untuk kombinasi ini di bawah ambang "
          f"({AMBANG_KEYAKINAN_SAMPEL}). Estimasi harga wajar & vonis likuiditas di atas "
          f"KEYAKINAN RENDAH -- gunakan sebagai referensi kasar saja, bukan acuan pasti.")
print("=" * 80 + "\n")