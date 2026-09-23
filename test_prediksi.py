"""
Script Uji Interaktif & Demonstrasi Klasifikasi Likuiditas Penjualan Mobil OLX
Pembaruan:
1. Memuat model klasifikasi (model_klasifikasi_penjualan.pkl).
2. Menerima input spesifikasi teknis + harga penawaran iklan.
3. Menerima input draf teks deskripsi untuk diekstraksi secara otomatis (tipe_penjual & garansi).
4. Menghasilkan probabilitas, label (Cepat vs Lambat), dan rekomendasi optimasi iklan.
"""

import os
import pickle
import pandas as pd
import numpy as np
import re
from datetime import datetime

# ============================================================
# 1. HELPER EXTRACTION UNTUK INPUT DESKRIPSI
# ============================================================
DAFTAR_MODEL = [
    "alphard", "vellfire", "avanza", "innova", "fortuner", "yaris", "rush",
    "calya", "agya", "raize", "corolla", "camry", "vios", "hilux", "sienta",
    "granace", "land cruiser", "hiace", "veloz", "harrier", "voxy",
    "brio", "hrv", "crv", "city", "civic", "jazz", "mobilio", "brv", "accord", "freed", "odyssey", "wrv",
    "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio",
    "xpander", "pajero", "outlander", "mirage", "triton", "eclipse",
    "ertiga", "xl7", "ignis", "baleno", "jimny", "karimun", "sx4", "s-presso", "grand vitara", "every",
    "grand livina", "livina", "serena", "xtrail", "juke", "march", "kicks", "magnite", "teana", "elgrand",
    "320i", "330i", "520i", "530i", "x1", "x3", "x5", "x7",
    "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "amg", "cla",
    "creta", "stargazer", "santa fe", "palisade", "ioniq", "tucson", "h-1",
    "confero", "almaz", "cortez", "air ev", "binguo", "alvez"
]

def ekstrak_fitur_dari_deskripsi(teks_deskripsi):
    teks = str(teks_deskripsi).lower()
    kw_dealer = ['showroom', 'paket kredit', 'tdp', 'dp ', 'angsuran', 'cicilan', 'leasing', 'bca finance', 'olxmobbi']
    tipe_penjual = 'Dealer' if any(k in teks for k in kw_dealer) else 'Individu'
    
    kw_garansi = ['garansi', 'warranty', 'sertifikat', 'otospector', 'lulus inspeksi', 'bebas banjir', 'bebas tabrak']
    ada_garansi = 'Ya' if any(k in teks for k in kw_garansi) else 'Tidak'
    
    return tipe_penjual, ada_garansi

# ============================================================
# 2. LOAD MODEL KLASIFIKASI
# ============================================================
file_model = "model_klasifikasi_penjualan.pkl"
if not os.path.exists(file_model):
    print(f"[!] File '{file_model}' belum ditemukan. Silakan jalankan script training klasifikasi terlebih dahulu.")
    exit()

with open(file_model, "rb") as f:
    pipeline_model = pickle.load(f)

# ============================================================
# 3. INTERFACE INPUT DATA USER
# ============================================================
print("=" * 65)
print("   AUTOVALUATE: PREDIKSI LIKUIDITAS PENJUALAN MOBIL BEKAS OLX   ")
print("=" * 65)

# Daftar Merek Populer
merek_pilihan = ["Toyota", "Honda", "Daihatsu", "Mitsubishi", "Suzuki", "Nissan", "Mercedes-Benz", "Bmw", "Hyundai", "Wuling"]
print("Pilih Merek Mobil:")
for i, m in enumerate(merek_pilihan, 1):
    print(f"[{i}] {m}", end="\t" if i % 4 != 0 else "\n")
print()

while True:
    try:
        idx_m = int(input("Masukkan nomor merek: ").strip())
        if 1 <= idx_m <= len(merek_pilihan):
            merek_input = merek_pilihan[idx_m - 1]
            break
        print("[!] Pilihan tidak valid.")
    except ValueError:
        print("[!] Masukkan angka yang sesuai.")

model_input = input("Masukkan Nama Seri/Model Mobil (contoh: Avanza, Innova, Brio): ").strip().title()

while True:
    try:
        tahun_input = int(input("Masukkan Tahun Pembuatan (2000 - 2026): ").strip())
        if 2000 <= tahun_input <= 2026:
            break
        print("[!] Tahun di luar batas wajar.")
    except ValueError:
        print("[!] Masukkan angka tahun yang valid.")

while True:
    trans_in = input("Transmisi (1. Otomatis / 2. Manual): ").strip()
    if trans_in in ['1', 'otomatis']:
        transmisi_input = 'otomatis'
        break
    elif trans_in in ['2', 'manual']:
        transmisi_input = 'manual'
        break
    print("[!] Ketik 1 atau 2.")

while True:
    try:
        km_input = int(input("Masukkan Odometer / Jarak Tempuh (KM): ").strip())
        if km_input >= 0:
            break
        print("[!] KM tidak boleh negatif.")
    except ValueError:
        print("[!] Masukkan angka jarak tempuh.")

while True:
    try:
        harga_input = float(input("Masukkan Rencana Harga Penawaran Iklan (Rp): ").strip())
        if harga_input > 0:
            break
        print("[!] Harga harus lebih dari 0.")
    except ValueError:
        print("[!] Masukkan nominal harga yang valid.")

# Input Deskripsi (Rekomendasi Dosen)
print("\nMasukkan Deskripsi Iklan (Bisa dikosongkan / tekan Enter jika tidak ada):")
deskripsi_input = input("> ").strip()

# ============================================================
# 4. PREPROCESSING & EKSTRAKSI FITUR DARI INPUT USER
# ============================================================
tipe_penjual_deteksi, garansi_deteksi = ekstrak_fitur_dari_deskripsi(deskripsi_input)
usia_mobil_hitung = max(1, 2026 - tahun_input)
km_per_tahun_hitung = int(round(km_input / usia_mobil_hitung))

data_uji = pd.DataFrame([{
    'merek': merek_input,
    'model': model_input,
    'transmisi': transmisi_input,
    'tipe_penjual': tipe_penjual_deteksi,
    'ada_garansi': garansi_deteksi,
    'harga': harga_input,
    'tahun': tahun_input,
    'jarak_tempuh': km_input,
    'usia_mobil': usia_mobil_hitung,
    'km_per_tahun': km_per_tahun_hitung
}])

# ============================================================
# 5. PREDIKSI MODEL & PENYUSUNAN REKOMENDASI
# ============================================================
prediksi_kelas = pipeline_model.predict(data_uji)[0]  # 1: Cepat, 0: Lambat
probabilitas = pipeline_model.predict_proba(data_uji)[0] if hasattr(pipeline_model, "predict_proba") else [0.5, 0.5]

prob_cepat = probabilitas[1] * 100
prob_lambat = probabilitas[0] * 100

if prediksi_kelas == 1:
    status_label = "[✓] CEPAT LAKU (< 30 HARI)"
    peluang = prob_cepat
    saran = (
        "Penetapan harga dan profil kendaraan Anda sangat atraktif bagi pasar. "
        "Unit memiliki peluang besar terjual cepat dalam kurun waktu 1 bulan pertama penayangan."
    )
else:
    status_label = "[!] LAMBAT LAKU (>= 30 HARI / POTENSI MACET)"
    peluang = prob_lambat
    saran = (
        "Unit berisiko tertahan lama di marketplace. "
        "Pertimbangkan menurunkan harga penawaran sebesar 5-8% atau cantumkan riwayat servis berkala, "
        "laporan inspeksi, serta opsi paket uang muka/garansi pada deskripsi untuk meningkatkan daya tarik pembeli."
    )

# ============================================================
# 6. TAMPILAN OUTPUT
# ============================================================
print("\n" + "=" * 65)
print("             HASIL ANALISIS DAYA JUAL & LIKUIDITAS              ")
print("=" * 65)
print(f"Kendaraan        : {merek_input} {model_input} ({tahun_input})")
print(f"Spesifikasi      : {transmisi_input.title()} | {km_input:,} KM ({km_per_tahun_hitung:,} KM/Tahun)")
print(f"Harga Penawaran  : Rp {int(harga_input):,}")
print("-" * 65)
print("HASIL EKSTRAKSI DARI DESKRIPSI (NLP/Text Matching):")
print(f"- Profil Penjual : {tipe_penjual_deteksi}")
print(f"- Garansi/Kondisi: {garansi_deteksi}")
print("-" * 65)
print(f"PREDIKSI MODEL   : {status_label}")
print(f"Tingkat Keyakinan: {peluang:.1f}%")
print("-" * 65)
print("REKOMENDASI SISTEM:")
print(saran)
print("=" * 65 + "\n")