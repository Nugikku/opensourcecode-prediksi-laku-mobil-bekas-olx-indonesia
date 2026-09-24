"""
AutoLiquid DSS: Decision Support System Valuasi & Likuiditas Mobil Bekas
Arsitektur: Hybrid (Machine Learning Regresi + Rule-Based Decision Engine)
"""

import os
import pickle
import pandas as pd
import numpy as np

FILE_MODEL = "model_regresi_harga.pkl"
FILE_DATA = "dataset_olx_bersih.csv"

# ============================================================
# 1. MEMUAT MODEL REGRESI & DATASET
# ============================================================
if not os.path.exists(FILE_MODEL) or not os.path.exists(FILE_DATA):
    print("[!] File model ('model_regresi_harga.pkl') atau dataset ('dataset_olx_bersih.csv') tidak ditemukan.")
    print("    Jalankan processing.py dan train_model_regresi.py terlebih dahulu.")
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

# ============================================================
# 2. PILIH SUDUT PANDANG PENGGUNA (PERSONA)
# ============================================================
print("Pilih Sudut Pandang Pengguna:")
print("1. Penjual (Evaluasi daya serap pasar & optimasi strategi harga)")
print("2. Pembeli (Evaluasi kewajaran deal, deteksi risiko & panduan negosiasi)")

while True:
    pilihan_role = input("Pilih (1/2): ").strip()
    if pilihan_role in ['1', '2']:
        is_penjual = (pilihan_role == '1')
        break
    print("[!] Masukkan angka 1 atau 2.")

# ============================================================
# 3. PILIH MEREK KENDARAAN (4 KOLOM SEJAJAR)
# ============================================================
print("\n" + "=" * 80)
print("PILIH MEREK KENDARAAN:")

daftar_merek = sorted(df_valid['merek'].unique())

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
        print("[!] Nomor merek tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka yang valid.")

# ============================================================
# 4. PILIH MODEL / SERI KENDARAAN
# ============================================================
df_merek = df_valid[df_valid['merek'] == merek_terpilih].copy()
daftar_model = sorted(df_merek['model'].unique())

print("\n" + "-" * 80)
print(f"Pilihan Model/Seri untuk {merek_terpilih}:")

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
        print("[!] Nomor model tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka yang valid.")

# ============================================================
# 5. MENAMPILKAN UNIT RIIL DARI MARKETPLACE
# ============================================================
df_unit_cocok = df_merek[df_merek['model'] == model_terpilih].copy().reset_index(drop=True)

print("\n" + "=" * 80)
print(f"DAFTAR UNIT AKTIF: {merek_terpilih.upper()} {model_terpilih.upper()} ({len(df_unit_cocok)} Unit Ditemukan)")
print("=" * 80)
print(f"{'No':<4} | {'Tahun':<6} | {'Transmisi':<10} | {'Jarak Tempuh':<14} | {'Harga Iklan':<16} | {'Penjual':<10}")
print("-" * 80)

for idx, row in df_unit_cocok.iterrows():
    harga_fmt = f"Rp {int(row['harga']):,}"
    km_fmt = f"{int(row['jarak_tempuh']):,} KM"
    penjual_fmt = str(row.get('tipe_penjual', 'Individu'))
    print(f"[{idx+1:<2}] | {int(row['tahun']):<6} | {str(row['transmisi']).title():<10} | {km_fmt:<14} | {harga_fmt:<16} | {penjual_fmt:<10}")

print("-" * 80)

while True:
    try:
        pilihan_unit = int(input(f"\nPilih nomor unit untuk dianalisis (1-{len(df_unit_cocok)}): ").strip())
        if 1 <= pilihan_unit <= len(df_unit_cocok):
            unit = df_unit_cocok.iloc[pilihan_unit - 1]
            break
        print("[!] Nomor unit tidak ada dalam tabel.")
    except ValueError:
        print("[!] Masukkan angka pilihan yang valid.")

# ============================================================
# 6. INFERENSI MACHINE LEARNING DENGAN PENGAMAN MINUS
# ============================================================
tahun_unit = int(unit['tahun'])
km_unit = int(unit['jarak_tempuh'])
usia_mobil = int(unit.get('usia_mobil', max(1, 2026 - tahun_unit)))
km_per_tahun = int(unit.get('km_per_tahun', round(km_unit / usia_mobil)))
transmisi_unit = str(unit['transmisi']).lower()
tipe_penjual = str(unit.get('tipe_penjual', 'Individu'))
ada_garansi = str(unit.get('ada_garansi', 'Tidak'))
harga_iklan = float(unit['harga'])
indikasi_tdp = int(unit.get('indikasi_kredit_tdp', 0))

data_uji = pd.DataFrame([{
    'merek': unit['merek'],
    'model': unit['model'],
    'transmisi': transmisi_unit,
    'tipe_penjual': tipe_penjual,
    'ada_garansi': ada_garansi,
    'tahun': tahun_unit,
    'jarak_tempuh': km_unit,
    'usia_mobil': usia_mobil,
    'km_per_tahun': km_per_tahun
}])

def hitung_harga_wajar_aman(data_row, unit_data):
    pred_mentah = float(model_pipeline.predict(data_row)[0])
    BATAS_MINIMUM = 20_000_000.0
    
    if pred_mentah < BATAS_MINIMUM:
        df_sejenis = df[(df['merek'] == unit_data['merek']) & (df['model'] == unit_data['model'])]
        if not df_sejenis.empty:
            return max(BATAS_MINIMUM, float(df_sejenis['harga'].median() * 0.55))
        return BATAS_MINIMUM
    return pred_mentah

harga_wajar_ml = hitung_harga_wajar_aman(data_uji, unit)
deviasi_persen = ((harga_iklan - harga_wajar_ml) / harga_wajar_ml) * 100

# ============================================================
# 7. RULE-BASED DECISION & LIQUIDITY ENGINE
# ============================================================
peringatan_anomali = None
if deviasi_persen <= -35.0 and indikasi_tdp == 1:
    peringatan_anomali = "Indikasi Jebakan TDP/Kredit (Bukan Total Harga Mobil Tunai)"

if deviasi_persen <= -3.0:
    if km_per_tahun > 30000 and ada_garansi == 'Tidak':
        status_likuiditas = "Sedang"
        alasan_likuiditas = "Harga di bawah pasar, namun kilometer tergolong tinggi tanpa jaminan garansi."
    else:
        status_likuiditas = "Cepat"
        alasan_likuiditas = "Harga penawaran sangat atraktif di bawah estimasi wajar pasar."
elif -3.0 < deviasi_persen <= 7.0:
    status_likuiditas = "Sedang"
    alasan_likuiditas = "Harga berada pada rentang wajar penyerapan pasar."
else:
    status_likuiditas = "Lambat"
    alasan_likuiditas = "Harga penawaran melebihi harga wajar pasar (overpriced)."

harga_target_cepat = harga_wajar_ml * 0.95
harga_target_sedang = harga_wajar_ml * 1.00
batas_nego_maksimal = harga_wajar_ml * 0.92

# ============================================================
# 8. OUTPUT SISTEM PENDUKUNG KEPUTUSAN
# ============================================================
print("\n" + "=" * 80)
print("                   AUTOLIQUID DSS: KEPUTUSAN PASAR                    ")
print("=" * 80)
print(f"Judul Iklan       : {unit.get('judul', '-')}")
print(f"Spesifikasi       : {unit['merek']} {unit['model']} ({tahun_unit}) | {transmisi_unit.title()}")
print(f"Odometer          : {km_unit:,} KM (Pemakaian: {km_per_tahun:,} KM/Tahun)")
print(f"Profil Penjual    : {tipe_penjual} | Jaminan Garansi: {ada_garansi}")
print("-" * 80)
print("VALUASI PASAR (MACHINE LEARNING CORE):")
print(f"- Harga Iklan Aktif       : Rp {int(harga_iklan):,}")
print(f"- Estimasi Harga Wajar ML : Rp {int(harga_wajar_ml):,}")
print(f"- Deviasi terhadap Wajar  : {deviasi_persen:+.1f}%")

if peringatan_anomali:
    print(f"\n[⚠️ PERINGATAN RISIKO] : {peringatan_anomali}")
    print("  Disarankan mengonfirmasi penjual apakah harga tertera adalah Total Tunai atau DP Kredit.")

print("-" * 80)
if status_likuiditas == "Cepat":
    print("STATUS LIKUIDITAS : [✓] CEPAT LAKU (< 15 Hari)")
elif status_likuiditas == "Sedang":
    print("STATUS LIKUIDITAS : [~] LAKU SEDANG (15 - 30 Hari)")
else:
    print("STATUS LIKUIDITAS : [!] LAMBAT LAKU (> 30 Hari / Tertahan)")

print(f"Analisis Penyerapan : {alasan_likuiditas}")
print("-" * 80)

print(f"REKOMENDASI SISTEM ({'SUDUT PANDANG PENJUAL' if is_penjual else 'SUDUT PANDANG PEMBELI'}):")
if is_penjual:
    if status_likuiditas == "Cepat":
        print("• Penetapan harga Anda sangat kompetitif dan berada dalam zona penyerapan cepat.")
        print("• Potensi unit terjual dalam hitungan hari sangat tinggi tanpa perlu penurunan harga.")
    elif status_likuiditas == "Sedang":
        print("• Harga penawaran Anda kompetitif di tingkat pasar wajar.")
        print(f"• 💡 SARAN CEPAT LAKU: Jika ingin mempercepat penjualan, sesuaikan ke Rp {int(harga_target_cepat):,}.")
    else:
        print("• Harga yang Anda pasang terlalu tinggi dibandingkan mobil sejenis.")
        print(f"• 💡 KOREKSI HARGA: Turunkan ke Rp {int(harga_target_sedang):,} (Wajar) atau Rp {int(harga_target_cepat):,} (Cepat Laku).")
else:
    if status_likuiditas == "Cepat":
        print("• Unit terindikasi sebagai 'Fair Deal / Best Offer'.")
        print("• Sangat menguntungkan bagi pembeli. Jadwalkan inspeksi fisik sebelum unit dibeli pihak lain.")
        print(f"• 💡 PANDUAN NEGO: Tawar tipis di kisaran Rp {int(batas_nego_maksimal):,}.")
    elif status_likuiditas == "Sedang":
        print("• Mobil ditawarkan pada harga wajar pasar.")
        print(f"• 💡 PANDUAN NEGO: Targetkan kesepakatan akhir di sekitar Rp {int(harga_target_cepat):,}.")
    else:
        print("• Unit terindikasi 'Overpriced' (kemahalan dibanding kondisi pasaran).")
        print(f"• 💡 PANDUAN NEGO: Hindari membeli di harga iklan; tekan negosiasi mendekati nilai wajar Rp {int(harga_wajar_ml):,}.")

print("=" * 80 + "\n")