"""
AutoLiquid DSS: Decision Support System Valuasi & Likuiditas Mobil Bekas
Arsitektur: Hybrid (Machine Learning Regresi + Rule-Based Decision Engine)
Mendukung Dual Mode: Input Mandiri Penjual & Eksplorasi Pembeli
"""

import os
import re
import pickle
import pandas as pd
import numpy as np

FILE_MODEL = "model_regresi_harga.pkl"
FILE_DATA = "dataset_olx_bersih.csv"

if not os.path.exists(FILE_MODEL) or not os.path.exists(FILE_DATA):
    print("[!] File model ('model_regresi_harga.pkl') atau dataset ('dataset_olx_bersih.csv') tidak ditemukan.")
    exit()

with open(FILE_MODEL, "rb") as f:
    model_pipeline = pickle.load(f)

df = pd.read_csv(FILE_DATA)
df_valid = df[
    (df['model'].fillna('').str.lower() != 'lainnya') & 
    (df['merek'].fillna('').str.lower() != 'lainnya') &
    (df['model'].fillna('').str.strip() != '') &
    (df['harga'].notna()) & 
    (df['tahun'].notna()) &
    (df['transmisi'].notna())
].copy().reset_index(drop=True)

print("=" * 80)
print("     AUTOLIQUID DSS: HYBRID PRICING & LIQUIDITY DECISION SUPPORT SYSTEM     ")
print("=" * 80)

print("Pilih Sudut Pandang Pengguna:")
print("1. Penjual (Input Mandiri Spesifikasi & Deskripsi Iklan)")
print("2. Pembeli (Analisis Iklan Marketplace yang Tersedia)")

while True:
    pilihan_role = input("Pilih (1/2): ").strip()
    if pilihan_role in ['1', '2']:
        is_penjual = (pilihan_role == '1')
        break
    print("[!] Masukkan angka 1 atau 2.")

# Seleksi Merek
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

# Seleksi Model
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

if is_penjual:
    print("\n" + "=" * 80)
    print("FORMULIR INPUT MANDIRI PENJUAL")
    print("=" * 80)
    tahun_unit = int(input("Masukkan Tahun Perakitan Mobil (1995-2026): ").strip())
    transmisi_unit = input("Transmisi (manual/automatic): ").strip().lower()
    km_unit = int(input("Masukkan Total Kilometer (KM): ").strip())
    harga_iklan = float(input("Masukkan Rencana Harga Jual (Rp): ").strip())
    tipe_penjual = "Individu"
    
    print("\nKetik Rencana Deskripsi Iklan:")
    deskripsi_teks = input("> ").strip().lower()

    # Ekstraksi NLP Real-Time
    pola_garansi = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|service record|buku servis|terawat)\b'
    pola_kredit = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit)\b'
    ada_garansi = 'Ya' if re.search(pola_garansi, deskripsi_teks) else 'Tidak'
    indikasi_tdp = 1 if re.search(pola_kredit, deskripsi_teks) else 0
    judul_iklan = "Unit Input Mandiri Penjual"
else:
    df_unit_cocok = df_merek[df_merek['model'] == model_terpilih].copy().reset_index(drop=True)
    print(f"\nDAFTAR UNIT AKTIF: {merek_terpilih.upper()} {model_terpilih.upper()}")
    print(f"{'No':<4} | {'Tahun':<6} | {'Transmisi':<10} | {'Jarak Tempuh':<14} | {'Harga Iklan':<16}")
    print("-" * 65)
    for idx, row in df_unit_cocok.iterrows():
        print(f"[{idx+1:<2}] | {int(row['tahun']):<6} | {str(row['transmisi']).title():<10} | {int(row['jarak_tempuh']):,} KM | Rp {int(row['harga']):,}")
    
    pilihan_u = int(input(f"\nPilih nomor unit (1-{len(df_unit_cocok)}): ").strip())
    unit = df_unit_cocok.iloc[pilihan_u - 1]
    
    tahun_unit = int(unit['tahun'])
    transmisi_unit = str(unit['transmisi']).lower()
    km_unit = int(unit['jarak_tempuh'])
    harga_iklan = float(unit['harga'])
    tipe_penjual = str(unit.get('tipe_penjual', 'Individu'))
    ada_garansi = str(unit.get('ada_garansi', 'Tidak'))
    indikasi_tdp = int(unit.get('indikasi_kredit_tdp', 0))
    judul_iklan = unit.get('judul', '-')

# Perhitungan Machine Learning & DSS
usia_mobil = max(1, 2026 - tahun_unit)
km_per_tahun = round(km_unit / usia_mobil)

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

pred_mentah = float(model_pipeline.predict(data_uji)[0])
BATAS_MINIMUM = 20_000_000.0

if pred_mentah < BATAS_MINIMUM:
    df_sejenis = df[(df['merek'] == merek_terpilih) & (df['model'] == model_terpilih)]
    harga_wajar_ml = max(BATAS_MINIMUM, float(df_sejenis['harga'].median() * 0.55)) if not df_sejenis.empty else BATAS_MINIMUM
else:
    harga_wajar_ml = pred_mentah

deviasi_persen = ((harga_iklan - harga_wajar_ml) / harga_wajar_ml) * 100

if deviasi_persen <= -3.0:
    status_likuiditas = "Cepat" if not (km_per_tahun > 30000 and ada_garansi == 'Tidak') else "Sedang"
elif -3.0 < deviasi_persen <= 7.0:
    status_likuiditas = "Sedang"
else:
    status_likuiditas = "Lambat"

print("\n" + "=" * 80)
print("                   AUTOLIQUID DSS: KEPUTUSAN PASAR                    ")
print("=" * 80)
print(f"Judul Unit        : {judul_iklan}")
print(f"Spesifikasi       : {merek_terpilih} {model_terpilih} ({tahun_unit}) | {transmisi_unit.title()}")
print(f"Odometer          : {km_unit:,} KM ({km_per_tahun:,} KM/Tahun)")
print(f"Deteksi Garansi   : {ada_garansi}")
print("-" * 80)
print(f"Harga Penawaran   : Rp {int(harga_iklan):,}")
print(f"Harga Wajar (ML)  : Rp {int(harga_wajar_ml):,}")
print(f"Deviasi Pasar     : {deviasi_persen:+.1f}%")
print(f"Vonis Likuiditas  : [{status_likuiditas.upper()} LAKU]")
print("=" * 80 + "\n")