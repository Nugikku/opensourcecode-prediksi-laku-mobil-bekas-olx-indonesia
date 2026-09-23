"""
Script Uji Interaktif & DSS Likuiditas Mobil Bekas (3 Kategori Likuiditas)
Alur Input:
1. Memilih Merek Mobil (Menu bernomor berkolom rapi).
2. Memilih Model/Tipe sesuai Merek yang dipilih.
3. Menampilkan seluruh daftar mobil riil dari dataset yang sesuai kriteria.
4. Memilih nomor unit untuk dianalisis (Persona Penjual vs Pembeli + Price Drop Simulator).
"""

import os
import pickle
import pandas as pd
import numpy as np

FILE_MODEL = "model_klasifikasi_penjualan.pkl"
FILE_DATA = "dataset_olx_bersih.csv"

# ============================================================
# 1. LOAD MODEL .PKL & DATASET RIIL
# ============================================================
if not os.path.exists(FILE_MODEL) or not os.path.exists(FILE_DATA):
    print("[!] File model atau dataset tidak ditemukan. Jalankan processing.py dan train_model_klasifikasi.py terlebih dahulu.")
    exit()

with open(FILE_MODEL, "rb") as f:
    model_pipeline = pickle.load(f)

df = pd.read_csv(FILE_DATA)

# Filter awal agar data yang disajikan tidak kosong dan bukan 'Lainnya'
df_valid = df[
    (df['model'].fillna('').str.lower() != 'lainnya') & 
    (df['merek'].fillna('').str.lower() != 'lainnya') &
    (df['model'].fillna('').str.strip() != '') &
    (df['harga'].notna()) & 
    (df['tahun'].notna()) &
    (df['transmisi'].notna())
].copy().reset_index(drop=True)

print("=" * 80)
print("     AUTOVALUATE: SMART LIQUIDITY & PRICING DECISION SYSTEM (3 KELAS)    ")
print("=" * 80)

# ============================================================
# 2. PILIH SUDUT PANDANG PENGGUNA (PERSONA)
# ============================================================
print("Pilih Sudut Pandang Pengguna:")
print("1. Penjual (Evaluasi kecepatan laku & simulasi optimasi harga)")
print("2. Pembeli (Evaluasi kelayakan deal & strategi tawar-menawar)")

while True:
    pilihan_role = input("Pilih (1/2): ").strip()
    if pilihan_role in ['1', '2']:
        is_penjual = (pilihan_role == '1')
        break
    print("[!] Masukkan 1 atau 2.")

# ============================================================
# 3. PILIH MEREK MOBIL (BERKOLOM RAPI)
# ============================================================
print("\n" + "=" * 80)
print("Pilih Merek Mobil:")

daftar_merek = sorted(df_valid['merek'].unique())

# Tampilkan merek dalam format 4 kolom sejajar
for i in range(0, len(daftar_merek), 4):
    baris_teks = ""
    for j in range(4):
        if i + j < len(daftar_merek):
            idx_nomor = i + j + 1
            nama_m = daftar_merek[i + j]
            baris_teks += f"[{idx_nomor:<2}] {nama_m:<16} "
    print(baris_teks)

while True:
    try:
        no_merek = int(input(f"\nPilih nomor merek (1-{len(daftar_merek)}): ").strip())
        if 1 <= no_merek <= len(daftar_merek):
            merek_terpilih = daftar_merek[no_merek - 1]
            break
        print("[!] Nomor tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka yang valid.")

# ============================================================
# 4. PILIH MODEL / TIPE SESUAI MEREK
# ============================================================
df_merek = df_valid[df_valid['merek'] == merek_terpilih].copy()
daftar_model = sorted(df_merek['model'].unique())

print("\n" + "-" * 80)
print(f"Pilihan Model/Seri untuk {merek_terpilih}:")

# Tampilkan model dalam format 4 kolom sejajar
for i in range(0, len(daftar_model), 4):
    baris_teks = ""
    for j in range(4):
        if i + j < len(daftar_model):
            idx_nomor = i + j + 1
            nama_mod = daftar_model[i + j]
            baris_teks += f"[{idx_nomor:<2}] {nama_mod:<16} "
    print(baris_teks)

while True:
    try:
        no_model = int(input(f"\nPilih nomor model (1-{len(daftar_model)}): ").strip())
        if 1 <= no_model <= len(daftar_model):
            model_terpilih = daftar_model[no_model - 1]
            break
        print("[!] Nomor tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka yang valid.")

# ============================================================
# 5. MENAMPILKAN SEMUA DATA RIIL YANG COCOK DARI DATASET
# ============================================================
df_unit_cocok = df_merek[df_merek['model'] == model_terpilih].copy().reset_index(drop=True)

print("\n" + "=" * 80)
print(f"DAFTAR SEMUA DATA RIIL UNTUK {merek_terpilih.upper()} {model_terpilih.upper()} ({len(df_unit_cocok)} Unit Ditemukan):")
print("=" * 80)
print(f"{'No':<4} | {'Tahun':<6} | {'Transmisi':<10} | {'Jarak Tempuh':<14} | {'Harga Iklan':<16} | {'Durasi Riil':<12}")
print("-" * 80)

for idx, row in df_unit_cocok.iterrows():
    harga_fmt = f"Rp {int(row['harga']):,}"
    km_fmt = f"{int(row['jarak_tempuh']):,} KM"
    durasi_fmt = f"{int(row.get('durasi_tayang_hari', 0))} hari"
    print(f"[{idx+1:<2}] | {int(row['tahun']):<6} | {str(row['transmisi']).title():<10} | {km_fmt:<14} | {harga_fmt:<16} | {durasi_fmt:<12}")

print("-" * 80)

while True:
    try:
        pilihan_unit = int(input(f"\nPilih nomor unit mobil yang ingin dianalisis (1-{len(df_unit_cocok)}): ").strip())
        if 1 <= pilihan_unit <= len(df_unit_cocok):
            unit = df_unit_cocok.iloc[pilihan_unit - 1]
            break
        print("[!] Nomor unit tidak ada di tabel.")
    except ValueError:
        print("[!] Masukkan angka pilihan yang valid.")

# ============================================================
# 6. EKSTRAKSI ATRIBUT RIIL & PREDIKSI MODEL
# ============================================================
merek = unit['merek']
model = unit['model']
tahun = int(unit['tahun'])
transmisi = str(unit['transmisi']).lower()
km = int(unit['jarak_tempuh'])
harga_iklan = float(unit['harga'])
usia_mobil = int(unit.get('usia_mobil', max(1, 2026 - tahun)))
km_per_tahun = int(unit.get('km_per_tahun', round(km / usia_mobil)))
tipe_penjual = str(unit.get('tipe_penjual', 'Individu'))
ada_garansi = str(unit.get('ada_garansi', 'Tidak'))
durasi_riil = int(unit.get('durasi_tayang_hari', 0))
status_riil = str(unit.get('kategori_penjualan', '-'))

# Median dan Deviasi Pasar Lokal
df_lokal = df[(df['model'] == model) & (df['tahun'] == tahun)]
median_pasar = df_lokal['harga'].median() if not df_lokal.empty else harga_iklan
deviasi_pasar = ((harga_iklan - median_pasar) / median_pasar) * 100

def buat_fitur(harga_tes, deviasi_tes):
    return pd.DataFrame([{
        'merek': merek,
        'model': model,
        'transmisi': transmisi,
        'tipe_penjual': tipe_penjual,
        'ada_garansi': ada_garansi,
        'harga': harga_tes,
        'tahun': tahun,
        'jarak_tempuh': km,
        'usia_mobil': usia_mobil,
        'km_per_tahun': km_per_tahun,
        'deviasi_harga_pasar': deviasi_tes
    }])

data_uji = buat_fitur(harga_iklan, deviasi_pasar)
prediksi_kelas = model_pipeline.predict(data_uji)[0]

# Probabilitas Keyakinan Model
if hasattr(model_pipeline, "predict_proba"):
    idx_kelas = list(model_pipeline.classes_).index(prediksi_kelas)
    prob_keyakinan = model_pipeline.predict_proba(data_uji)[0][idx_kelas] * 100
else:
    prob_keyakinan = 85.0

# ============================================================
# 7. SIMULASI PENURUNAN HARGA (PRICE DROP SIMULATOR)
# ============================================================
harga_rekomendasi = harga_iklan
if prediksi_kelas in ['Lambat', 'Sedang']:
    for i in range(1, 20):  # Penurunan bertahap sampai 40%
        harga_simulasi = harga_iklan * (1 - (0.02 * i))
        deviasi_simulasi = ((harga_simulasi - median_pasar) / median_pasar) * 100
        if model_pipeline.predict(buat_fitur(harga_simulasi, deviasi_simulasi))[0] == 'Cepat':
            harga_rekomendasi = harga_simulasi
            break

# ============================================================
# 8. OUTPUT KEPUTUSAN SISTEM RIIL
# ============================================================
print("\n" + "=" * 80)
print("                   HASIL ANALISIS SISTEM KEPUTUSAN RIIL                   ")
print("=" * 80)
print(f"Judul Iklan Riil  : {unit.get('judul', '-')}")
print(f"Spesifikasi       : {merek} {model} ({tahun}) | Transmisi: {transmisi.title()}")
print(f"Odometer          : {km:,} KM (Rata-rata: {km_per_tahun:,} KM/Tahun)")
print(f"Profil Penjual    : {tipe_penjual} | Status Garansi: {ada_garansi}")
print("-" * 80)
print("RIWAYAT TAYANG & DATA PASAR:")
print(f"- Tanggal Posting : {unit.get('tanggal_posting', 'Tersedia di sistem')}")
print(f"- Durasi Tayang   : {durasi_riil} Hari di Marketplace")
print(f"- Status Asli     : Kategori '{status_riil}'")
print(f"- Harga Iklan     : Rp {int(harga_iklan):,}")
print(f"- Median Pasar    : Rp {int(median_pasar):,} (Deviasi: {deviasi_pasar:+.1f}%)")
print("-" * 80)

if prediksi_kelas == 'Cepat':
    status_label = "[✓] CEPAT LAKU (< 15 Hari)"
elif prediksi_kelas == 'Sedang':
    status_label = "[~] LAKU SEDANG (15 - 30 Hari)"
else:
    status_label = "[!] LAMBAT LAKU (> 30 Hari / Rawan Macet)"

print(f"PREDIKSI MODEL    : {status_label}")
print(f"Tingkat Keyakinan : {prob_keyakinan:.2f}%")
print("-" * 80)

print(f"KESIMPULAN & REKOMENDASI ({'SUDUT PANDANG PENJUAL' if is_penjual else 'SUDUT PANDANG PEMBELI'}):")
if is_penjual:
    if prediksi_kelas == 'Cepat':
        print("• Penetapan harga iklan Anda sangat atraktif dan sesuai daya serap pasar.")
        print("• Unit diprediksi cepat diminati dan terjual dalam 2 minggu pertama.")
    elif prediksi_kelas == 'Sedang':
        print("• Harga berada pada batas rata-rata pasar; iklan membutuhkan waktu penyerapan wajar.")
        print(f"• 💡 OPTIMASI: Agar beralih ke 'Cepat Laku', pertimbangkan menyesuaikan harga ke Rp {int(harga_rekomendasi):,}.")
    else:
        print("• Harga dinilai terlalu tinggi dibanding kilometer dan usia mobil, rawan mengendap lama.")
        print(f"• 💡 KOREKSI HARGA: Turunkan harga ke sekitar Rp {int(harga_rekomendasi):,} agar unit terserap pasar.")
else:
    if prediksi_kelas == 'Cepat':
        print("• Unit ini berstatus 'Fair Deal / Best Price' (harga kompetitif dibanding pasar).")
        print("• Sangat layak dibeli. Segera lakukan inspeksi sebelum unit terjual ke calon pembeli lain.")
    elif prediksi_kelas == 'Sedang':
        print("• Unit dijual dengan harga standar pasar.")
        print(f"• 💡 STRATEGI NEGO: Ajukan penawaran awal di kisaran Rp {int(harga_rekomendasi):,} untuk mendapatkan deal terbaik.")
    else:
        print("• Unit terindikasi 'Overpriced' (kemahalan) terhadap tren kondisi pasaran.")
        print(f"• 💡 STRATEGI NEGO: Gunakan lama waktu tayang iklan sebagai bahan tawar, patok batas maksimal di Rp {int(harga_rekomendasi):,}.")

print("=" * 80 + "\n")