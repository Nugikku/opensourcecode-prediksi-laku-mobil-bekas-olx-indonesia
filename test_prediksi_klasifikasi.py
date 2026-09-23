import os
import pickle
import pandas as pd
import numpy as np

FILE_MODEL = "model_klasifikasi_penjualan.pkl"
FILE_DATA = "dataset_olx_bersih.csv"

# ============================================================
# 1. LOAD MODEL FILE .PKL
# ============================================================
if not os.path.exists(FILE_MODEL):
    print(f"\n[!] File '{FILE_MODEL}' tidak ditemukan.")
    print("[!] Jalankan file 'train_model_klasifikasi.py' terlebih dahulu.\n")
    exit()

if not os.path.exists(FILE_DATA):
    print(f"\n[!] File '{FILE_DATA}' tidak ditemukan.\n")
    exit()

# Membaca model klasifikasi tersimpan (.pkl)
with open(FILE_MODEL, "rb") as f:
    model_pipeline = pickle.load(f)

df = pd.read_csv(FILE_DATA)

# Memastikan kolom fitur pendukung tersedia
if 'usia_mobil' not in df.columns:
    df['usia_mobil'] = 2026 - df['tahun']
if 'km_per_tahun' not in df.columns:
    df['km_per_tahun'] = (df['jarak_tempuh'] / df['usia_mobil'].apply(lambda x: max(1, x))).round(0).astype(int)
if 'tipe_penjual' not in df.columns:
    deskripsi_teks = df['deskripsi'].fillna('').str.lower()
    df['tipe_penjual'] = deskripsi_teks.apply(
        lambda t: 'Dealer' if any(k in t for k in ['showroom', 'paket kredit', 'tdp', 'dp ', 'otospector', 'olxmobbi']) else 'Individu'
    )
if 'ada_garansi' not in df.columns:
    deskripsi_teks = df['deskripsi'].fillna('').str.lower()
    df['ada_garansi'] = deskripsi_teks.apply(
        lambda t: 'Ya' if any(k in t for k in ['garansi', 'warranty', 'sertifikat', 'otospector']) else 'Tidak'
    )

# ============================================================
# 2. TAMPILKAN 10 DAFTAR PILIHAN MOBIL
# ============================================================
print("=" * 80)
print("     AUTOVALUATE: UJI KLASIFIKASI LIKUIDITAS PENJUALAN MOBIL OLX     ")
print("=" * 80)

df_sampel = df.sample(n=10, random_state=42).reset_index(drop=True)

print(f"{'No':<4} | {'Merek':<12} | {'Model':<12} | {'Tahun':<6} | {'Transmisi':<10} | {'Harga (Rp)':<16} | {'Status Riil':<10}")
print("-" * 80)

for idx, row in df_sampel.iterrows():
    harga_fmt = f"Rp {int(row['harga']):,}"
    status_riil = row.get('kategori_penjualan', '-')
    print(f"[{idx+1:<2}] | {str(row['merek']):<12} | {str(row['model']):<12} | {int(row['tahun']):<6} | {str(row['transmisi']):<10} | {harga_fmt:<16} | {status_riil:<10}")

print("-" * 80)

# ============================================================
# 3. MEMILIH NOMOR MOBIL
# ============================================================
while True:
    try:
        pilihan = int(input(f"\nPilih nomor mobil (1-{len(df_sampel)}): ").strip())
        if 1 <= pilihan <= len(df_sampel):
            unit = df_sampel.iloc[pilihan - 1]
            break
        print("[!] Nomor tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka pilihan yang valid.")

# ============================================================
# 4. PREDIKSI MENGGUNAKAN PIPELINE .PKL
# ============================================================
fitur_wajib = [
    'merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi',
    'harga', 'tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun'
]

data_input = pd.DataFrame([unit[fitur_wajib]])

# Eksekusi prediksi via model .pkl
prediksi_nilai = model_pipeline.predict(data_input)[0]  # 1: Cepat, 0: Lambat
probabilitas = model_pipeline.predict_proba(data_input)[0] if hasattr(model_pipeline, "predict_proba") else [0.5, 0.5]

prob_cepat = probabilitas[1] * 100
prob_lambat = probabilitas[0] * 100

if prediksi_nilai == 1:
    label_prediksi = "[✓] CEPAT LAKU (< 30 HARI)"
    skor_keyakinan = prob_cepat
    saran_sistem = (
        "Penetapan harga dan profil kilometer unit ini sangat kompetitif di pasar. "
        "Mobil memiliki probabilitas tinggi untuk terjual dalam kurun waktu kurang dari 30 hari."
    )
else:
    label_prediksi = "[!] LAMBAT LAKU (>= 30 HARI / RAWAN MACET)"
    skor_keyakinan = prob_lambat
    saran_sistem = (
        "Unit berisiko tertahan lama di marketplace. "
        "Disarankan untuk menurunkan harga penawaran sekitar 5-8% atau menyertakan garansi mesin "
        "serta kelengkapan bukti servis berkala pada deskripsi iklan."
    )

# ============================================================
# 5. TAMPILAN KEPUTUSAN PREDIKSI
# ============================================================
print("\n" + "=" * 80)
print("                       HASIL ANALISIS PREDIKSI                        ")
print("=" * 80)
print(f"Judul Iklan      : {unit.get('judul', '-')}")
print(f"Spesifikasi      : {unit['merek']} {unit['model']} ({int(unit['tahun'])})")
print(f"Transmisi / Odo  : {str(unit['transmisi']).title()} | {int(unit['jarak_tempuh']):,} KM ({int(unit['km_per_tahun']):,} KM/Tahun)")
print(f"Harga Penawaran  : Rp {int(unit['harga']):,}")
print("-" * 80)
print("ATRIBUT DESKRIPSI (NLP TEXT MINING):")
print(f"- Profil Penjual : {unit.get('tipe_penjual', 'Individu')}")
print(f"- Status Garansi : {unit.get('ada_garansi', 'Tidak')}")
print("-" * 80)
print(f"PREDIKSI MODEL   : {label_prediksi}")
print(f"Tingkat Keyakinan: {skor_keyakinan:.2f}%")
if 'kategori_penjualan' in unit:
    print(f"Status Data Riil : {unit['kategori_penjualan']}")
print("-" * 80)
print("REKOMENDASI SISTEM:")
print(saran_sistem)
print("=" * 80 + "\n")