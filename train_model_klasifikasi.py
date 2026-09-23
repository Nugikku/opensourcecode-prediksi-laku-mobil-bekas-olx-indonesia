"""
Script Uji Interaktif Klasifikasi Likuiditas Penjualan Mobil OLX
Mode: Memilih unit langsung dari dataset yang sudah ada (Tanpa ketik manual).
"""

import os
import pickle
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD MODEL KLASIFIKASI & DATASET BERSIH
# ============================================================
file_model = "model_klasifikasi_penjualan.pkl"
file_data = "dataset_olx_bersih.csv"

if not os.path.exists(file_model):
    raise FileNotFoundError(f"Model '{file_model}' belum ditemukan. Jalankan train_model_klasifikasi.py terlebih dahulu.")

if not os.path.exists(file_data):
    raise FileNotFoundError(f"File '{file_data}' tidak ditemukan.")

with open(file_model, "rb") as f:
    model_pipeline = pickle.load(f)

df = pd.read_csv(file_data)

# ============================================================
# 2. SELEKSI DATA SAMPEL DARI DATASET RIIL
# ============================================================
print("=" * 70)
print("   AUTOVALUATE: UJI KLASIFIKASI LIKUIDITAS DARI DATASET RIIL   ")
print("=" * 70)

# Ambil 10 sampel acak dari dataset untuk dipilih pengguna
df_sampel = df.sample(n=10, random_state=42).reset_index(drop=True)

print("Pilih salah satu mobil dari dataset berikut untuk dianalisis:\n")
print(f"{'No':<4} | {'Merek':<12} | {'Model':<12} | {'Tahun':<6} | {'Transmisi':<10} | {'Harga (Rp)':<16} | {'Status Asli':<10}")
print("-" * 75)

for idx, row in df_sampel.iterrows():
    harga_fmt = f"Rp {int(row['harga']):,}"
    status_asli = row.get('kategori_penjualan', '-')
    print(f"[{idx+1:<2}] | {str(row['merek']):<12} | {str(row['model']):<12} | {int(row['tahun']):<6} | {str(row['transmisi']):<10} | {harga_fmt:<16} | {status_asli:<10}")

print("-" * 75)

# ============================================================
# 3. PEMILIHAN OLEH PENGGUNA (NOMOR PILIHAN)
# ============================================================
while True:
    try:
        pilihan = int(input(f"\nMasukkan nomor mobil yang ingin dianalisis (1-{len(df_sampel)}): ").strip())
        if 1 <= pilihan <= len(df_sampel):
            unit_terpilih = df_sampel.iloc[pilihan - 1]
            break
        print("[!] Nomor tidak tersedia dalam daftar.")
    except ValueError:
        print("[!] Masukkan angka yang valid.")

# ============================================================
# 4. PREDIKSI MODEL & PROBABILITAS
# ============================================================
fitur_wajib = [
    'merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi',
    'harga', 'tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun'
]

# Bentuk satu baris dataframe sesuai skema input model
data_uji = pd.DataFrame([unit_terpilih[fitur_wajib]])

prediksi_kelas = model_pipeline.predict(data_uji)[0]  # 1: Cepat, 0: Lambat
probabilitas = model_pipeline.predict_proba(data_uji)[0] if hasattr(model_pipeline, "predict_proba") else [0.5, 0.5]

prob_cepat = probabilitas[1] * 100
prob_lambat = probabilitas[0] * 100

if prediksi_kelas == 1:
    label_hasil = "[✓] CEPAT LAKU (< 30 HARI)"
    keyakinan = prob_cepat
    rekomendasi = (
        "Harga penawaran dan riwayat kilometer unit ini berada dalam rentang ideal pasar. "
        "Unit memiliki probabilitas tinggi untuk langsung terjual dalam kurun waktu kurang dari 1 bulan."
    )
else:
    label_hasil = "[!] LAMBAT LAKU (>= 30 HARI / RAWAN MACET)"
    keyakinan = prob_lambat
    rekomendasi = (
        "Unit berpotensi tertahan lama di marketplace karena kombinasi harga penawaran dan profil kilometer. "
        "Disarankan untuk menurunkan harga penawaran sekitar 5-10% atau menyertakan garansi mesin "
        "serta sertifikat inspeksi kendaraan pada deskripsi iklan."
    )

# ============================================================
# 5. TAMPILAN OUTPUT DETAIL KEPUTUSAN
# ============================================================
print("\n" + "=" * 70)
print("                    HASIL PREDIKSI LIKUIDITAS                  ")
print("=" * 70)
print(f"Judul Iklan Asli : {unit_terpilih.get('judul', '-')}")
print(f"Spesifikasi Unit : {unit_terpilih['merek']} {unit_terpilih['model']} ({int(unit_terpilih['tahun'])})")
print(f"Transmisi / KM   : {str(unit_terpilih['transmisi']).title()} | {int(unit_terpilih['jarak_tempuh']):,} KM")
print(f"Harga Iklan      : Rp {int(unit_terpilih['harga']):,}")
print("-" * 70)
print("FITUR EKSTRAKSI DARI DESKRIPSI (TEXT MINING):")
print(f"- Profil Penjual : {unit_terpilih.get('tipe_penjual', 'Individu')}")
print(f"- Status Garansi : {unit_terpilih.get('ada_garansi', 'Tidak')}")
print("-" * 70)
print(f"PREDIKSI MODEL   : {label_hasil}")
print(f"Tingkat Keyakinan: {keyakinan:.2f}%")
print(f"Status Data Asli : {unit_terpilih.get('kategori_penjualan', '-')}")
print("-" * 70)
print("REKOMENDASI SISTEM:")
print(rekomendasi)
print("=" * 70 + "\n")